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

//! Scaled decoding tests: `Decoder::nth_image_with_options()` and
//! `Decoder::next_image_with_options()` with `scaled_size` (both alone and
//! combined with `crop_rect`) are compared against a full decode followed by
//! an optional crop and a downscale (`Image::scale`), across codecs, pixel
//! formats, bit depths, clean apertures, overlays, sample transforms, gain
//! maps, thumbnails, animations and grid configurations.

#![allow(clippy::disallowed_methods, clippy::disallowed_macros)]

use crabby_avif::decoder::DecodeOptions;
use crabby_avif::decoder::ImageContentType;
use crabby_avif::decoder::ImageSize;
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
        || !cfg!(feature = "libyuv")
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

fn size(width: u32, height: u32) -> ImageSize {
    ImageSize { width, height }
}

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

fn base_clap_rect(image: &Image) -> CropRect {
    let full = crop(0, 0, image.width, image.height);
    match &image.clap {
        Some(clap) => {
            CropRect::create_from(clap, image.width, image.height, image.yuv_format).unwrap_or(full)
        }
        None => full,
    }
}

fn reference_crop_and_scale(
    ref_full: &Image,
    crop_opt: Option<CropRect>,
    scaled_opt: Option<ImageSize>,
) -> AvifResult<Image> {
    if crop_opt.is_none() && scaled_opt.is_none() {
        return ref_full.try_deep_clone();
    }
    let base = base_clap_rect(ref_full);
    let effective_rect = match crop_opt {
        Some(mut r) => {
            r.x += base.x;
            r.y += base.y;
            r
        }
        None => base,
    };
    let covers_full = effective_rect.x == 0
        && effective_rect.y == 0
        && effective_rect.width == ref_full.width
        && effective_rect.height == ref_full.height;
    let cropped = if covers_full {
        let mut copy = ref_full.try_deep_clone()?;
        copy.clap = None;
        copy
    } else {
        crop_reference_image(ref_full, &effective_rect)?
    };
    match scaled_opt {
        Some(sz) => scale_reference_image(&cropped, sz.width, sz.height),
        None => Ok(cropped),
    }
}

// Candidate target sizes for scaling down a `w x h` region: identity, halves,
// quarters, thirds, odd/prime non-divisors, anisotropic aspect ratios, and 1x1.
fn scaled_sizes_for_non_grid(w: u32, h: u32) -> Vec<ImageSize> {
    let candidates = [
        size(w, h),
        size(w.div_ceil(2), h.div_ceil(2)),
        size((w / 2).max(1), (h / 2).max(1)),
        size((w / 3).max(1), (h / 3).max(1)),
        size((w / 4).max(1), (h / 4).max(1)),
        size(((w * 2) / 3).max(1) | 1, ((h * 2) / 5).max(1) | 1),
        size((w / 2).max(1), h),
        size(w, (h / 2).max(1)),
        size(2.min(w), 2.min(h)),
        size(1, 1),
    ];
    let mut out = Vec::new();
    for s in candidates {
        if s.width > 0 && s.height > 0 && s.width <= w && s.height <= h && !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

fn check_non_grid_exact(
    data: &[u8],
    codec: CodecChoice,
    gainmap: bool,
    allow_sample_transform: bool,
    loose_strictness: bool,
    index: u32,
    name: &str,
) -> AvifResult<()> {
    let mut ref_decoder = decoder_for_data(data, codec, gainmap);
    ref_decoder.settings.allow_sample_transform = allow_sample_transform;
    if loose_strictness {
        ref_decoder.settings.strictness = decoder::Strictness::None;
    }
    ref_decoder.parse()?;
    ref_decoder.nth_image(index)?;
    let ref_full = decoded_image(&ref_decoder, gainmap).try_deep_clone()?;
    let base = base_clap_rect(&ref_full);
    let (bw, bh) = (base.width, base.height);

    let mut decoder = decoder_for_data(data, codec, gainmap);
    decoder.settings.allow_sample_transform = allow_sample_transform;
    if loose_strictness {
        decoder.settings.strictness = decoder::Strictness::None;
    }
    decoder.parse()?;

    // 1. Scaling without crop_rect.
    for sz in scaled_sizes_for_non_grid(bw, bh) {
        let opts = DecodeOptions {
            crop_rect: None,
            scaled_size: Some(sz),
        };
        decoder
            .nth_image_with_options(index, &opts)
            .unwrap_or_else(|err| panic!("{name}: scale {sz:?} failed: {err:?}"));
        let actual = decoded_image(&decoder, gainmap);
        let expected = reference_crop_and_scale(&ref_full, None, Some(sz))?;
        assert_eq!(
            (actual.width, actual.height),
            (sz.width, sz.height),
            "{name}"
        );
        assert!(actual.clap.is_none(), "{name}");
        assert_eq!(actual.irot_angle, ref_full.irot_angle, "{name}");
        assert_eq!(actual.imir_axis, ref_full.imir_axis, "{name}");
        assert!(
            are_images_equal(actual, &expected)?,
            "{name}: scale {sz:?} ({codec:?}) differs from full decode + scale"
        );
        let actual_rgba = to_rgba(actual)?;
        let expected_rgba = to_rgba(&expected)?;
        assert!(
            are_rgb_images_equal(&actual_rgba, &expected_rgba)?,
            "{name}: RGBA of scale {sz:?} ({codec:?}) differs"
        );
    }

    // 2. Combined crop_rect + scaled_size.
    let crop_candidates = [
        crop(0, 0, bw, bh),
        crop(
            1.min(bw - 1),
            1.min(bh - 1),
            (bw - 1).max(1),
            (bh - 1).max(1),
        ),
        crop(
            (bw / 4) | (1.min(bw.saturating_sub(1))),
            (bh / 4) | (1.min(bh.saturating_sub(1))),
            (bw / 2).max(1),
            (bh / 2).max(1),
        ),
        crop(60, 61, 25, 19),
    ];
    for c in crop_candidates {
        if c.width == 0 || c.height == 0 || c.x + c.width > bw || c.y + c.height > bh {
            continue;
        }
        for sz in [
            size(c.width, c.height),
            size((c.width / 2).max(1), (c.height / 2).max(1)),
            size((c.width / 3).max(1), ((c.height * 2) / 3).max(1)),
            size(1, 1),
        ] {
            let opts = DecodeOptions {
                crop_rect: Some(c),
                scaled_size: Some(sz),
            };
            decoder
                .nth_image_with_options(index, &opts)
                .unwrap_or_else(|err| panic!("{name}: crop {c:?} + scale {sz:?} failed: {err:?}"));
            let actual = decoded_image(&decoder, gainmap);
            let expected = reference_crop_and_scale(&ref_full, Some(c), Some(sz))?;
            assert_eq!(
                (actual.width, actual.height),
                (sz.width, sz.height),
                "{name}"
            );
            assert!(actual.clap.is_none(), "{name}");
            assert!(
                are_images_equal(actual, &expected)?,
                "{name}: crop {c:?} + scale {sz:?} ({codec:?}) differs"
            );
        }
    }

    // 3. Subsequent full decode restores original image and properties.
    decoder.nth_image(index)?;
    assert!(
        are_images_equal(decoded_image(&decoder, gainmap), &ref_full)?,
        "{name}: full decode after scaled decodes differs"
    );
    Ok(())
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

fn region_decode_file(name: &str) -> Vec<u8> {
    std::fs::read(get_test_file(&format!("region_decode/{name}")))
        .unwrap_or_else(|err| panic!("Unable to read {name}: {err}"))
}

// Non-grid images across all YUV formats (400/420/422/444), bit depths (8/10/12),
// alpha modes, and odd/even/1x1 dimensions must match full decode (+ crop) + scale
// bit-for-bit.
#[test_matrix(
    [PixelFormat::Yuv400, PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444],
    [8, 10, 12],
    [false, true],
    [DAV1D, NATIVE]
)]
fn scale_decode_non_grid_formats(
    yuv_format: PixelFormat,
    depth: u8,
    alpha: bool,
    codec: CodecChoice,
) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let yuv = yuv_format_name(yuv_format);
    let alpha_suffix = if alpha { "_alpha" } else { "" };
    for dims in ["301x203", "64x64", "1x1", "2x3", "129x66"] {
        let name = format!("formats_{yuv}_{depth}bpc{alpha_suffix}_{dims}.avif");
        let data = region_decode_file(&name);
        check_non_grid_exact(&data, codec, false, false, false, 0, &name)?;
    }
    Ok(())
}

// Non-grid special features: clean aperture + rotation + mirroring, 10bpc,
// premultiplied alpha, gain maps, overlays ('iovl'), and 16-bit sample transforms ('sato').
#[test_matrix([DAV1D, NATIVE])]
fn scale_decode_non_grid_special_features(codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    for (filename, gainmap, allow_sato, loose) in [
        ("region/region_clap_irot_imir.avif", false, false, false),
        ("paris_10bpc.avif", false, false, false),
        ("alpha.avif", false, false, false),
        ("alpha_premultiplied.avif", false, false, false),
        ("seine_sdr_gainmap_srgb.avif", false, false, false),
        ("seine_sdr_gainmap_srgb.avif", true, false, false),
        ("overlay_with_border.avif", false, false, true),
        ("overlay_yellow_bg.avif", false, false, true),
        ("weld_sato_8plus8bit.avif", false, true, false),
        ("weld_sato_8plus8bit_alpha.avif", false, true, false),
        ("weld_sato_12plus4bit.avif", false, true, false),
    ] {
        let data = std::fs::read(get_test_file(filename)).expect("Unable to read file");
        check_non_grid_exact(&data, codec, gainmap, allow_sato, loose, 0, filename)?;
    }

    // For gainmap-only scaled decode, verify that the color image stays unscaled and has no planes.
    let gm_data =
        std::fs::read(get_test_file("seine_sdr_gainmap_srgb.avif")).expect("Unable to read file");
    let mut gm_dec = decoder_for_data(&gm_data, codec, true);
    gm_dec.parse()?;
    let orig_color_dims = (
        gm_dec.image().unwrap().width,
        gm_dec.image().unwrap().height,
    );
    gm_dec.nth_image_with_options(
        0,
        &DecodeOptions {
            crop_rect: None,
            scaled_size: Some(size(100, 75)),
        },
    )?;
    assert_eq!(
        (
            gm_dec.image().unwrap().width,
            gm_dec.image().unwrap().height
        ),
        orig_color_dims
    );
    assert!(!gm_dec.image().unwrap().has_plane(Plane::Y));
    assert_eq!(
        (gm_dec.gainmap().image.width, gm_dec.gainmap().image.height),
        (100, 75)
    );

    // Clean aperture non-grid: scaled_size exceeding clap width (96) must fail even when <= 120.
    let clap_data =
        std::fs::read(get_test_file("region/region_clap_irot_imir.avif")).expect("read clap");
    let mut clap_dec = decoder_for_data(&clap_data, codec, false);
    clap_dec.parse()?;
    assert_eq!(
        clap_dec.nth_image_with_options(
            0,
            &DecodeOptions {
                crop_rect: None,
                scaled_size: Some(size(100, 100)),
            },
        ),
        Err(AvifError::InvalidArgument)
    );
    Ok(())
}

// Non-grid frame whose 'ispe' differs from the coded AV1 frame dimensions.
#[test_matrix([(150, 90), (260, 150)], [DAV1D, NATIVE])]
fn scale_decode_scaled_ispe_frame(ispe: (u32, u32), codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let mut data = region_decode_file("scaled_200x120.avif");
    let mut patched = 0;
    for offset in 0..data.len() - 16 {
        if &data[offset..offset + 4] == b"ispe" {
            data[offset + 8..offset + 12].copy_from_slice(&ispe.0.to_be_bytes());
            data[offset + 12..offset + 16].copy_from_slice(&ispe.1.to_be_bytes());
            patched += 1;
        }
    }
    assert_eq!(patched, 1);
    check_non_grid_exact(&data, codec, false, false, false, 0, "scaled_ispe")
}

// Animated sequences: keyframes on both DAV1D and NATIVE, plus all inter-frames on DAV1D.
#[test_matrix([DAV1D, NATIVE])]
fn scale_decode_sequence(codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) || !cfg!(feature = "dav1d") {
        return Ok(());
    }
    let data = std::fs::read(get_test_file("colors-animated-12bpc-keyframes-0-2-3.avif"))
        .expect("Unable to read file");
    let mut ref_decoder = decoder_for_data(&data, DAV1D, false);
    ref_decoder.parse()?;
    let mut ref_frames = Vec::new();
    for i in 0..ref_decoder.image_count() {
        ref_decoder.nth_image(i)?;
        ref_frames.push(ref_decoder.image().unwrap().try_deep_clone()?);
    }
    let (w, h) = (ref_frames[0].width, ref_frames[0].height);
    let mut decoder = decoder_for_data(&data, codec, false);
    decoder.parse()?;
    for &index in &[3u32, 0, 2, 2, 0, 3] {
        for opts in [
            DecodeOptions {
                crop_rect: None,
                scaled_size: Some(size((w / 2).max(1), (h / 2).max(1))),
            },
            DecodeOptions {
                crop_rect: Some(crop(3, 5, w / 2, h / 2)),
                scaled_size: Some(size((w / 4).max(1), (h / 4).max(1))),
            },
        ] {
            decoder.nth_image_with_options(index, &opts)?;
            let expected = reference_crop_and_scale(
                &ref_frames[index as usize],
                opts.crop_rect,
                opts.scaled_size,
            )?;
            assert!(are_images_equal(decoder.image().unwrap(), &expected)?);
        }
    }

    // Full sequence including inter-frames with DAV1D (sequential + random access).
    if codec == DAV1D {
        for filename in [
            "colors-animated-8bpc.avif",
            "colors-animated-8bpc-alpha-exif-xmp.avif",
        ] {
            let seq_data = std::fs::read(get_test_file(filename)).expect("Unable to read file");
            let mut seq_ref = decoder_for_data(&seq_data, DAV1D, false);
            seq_ref.parse()?;
            let mut seq_dec = decoder_for_data(&seq_data, DAV1D, false);
            seq_dec.parse()?;
            let (sw, sh) = (
                seq_ref.image().unwrap().width,
                seq_ref.image().unwrap().height,
            );
            let opts = DecodeOptions {
                crop_rect: Some(crop(4, 6, sw - 8, sh - 12)),
                scaled_size: Some(size((sw - 8) / 2, (sh - 12) / 2)),
            };
            for i in 0..seq_ref.image_count() {
                seq_ref.next_image()?;
                seq_dec.next_image_with_options(&opts)?;
                let expected = reference_crop_and_scale(
                    seq_ref.image().unwrap(),
                    opts.crop_rect,
                    opts.scaled_size,
                )?;
                assert!(
                    are_images_equal(seq_dec.image().unwrap(), &expected)?,
                    "{filename} frame {i}"
                );
            }
            for &idx in &[4u32, 1, 3, 0, 2] {
                seq_ref.nth_image(idx)?;
                seq_dec.nth_image_with_options(idx, &opts)?;
                let expected = reference_crop_and_scale(
                    seq_ref.image().unwrap(),
                    opts.crop_rect,
                    opts.scaled_size,
                )?;
                assert!(
                    are_images_equal(seq_dec.image().unwrap(), &expected)?,
                    "{filename} nth frame {idx}"
                );
            }
        }
    }
    Ok(())
}

// Helper verifying cell-level downscaling on a grid image against full decode (+ crop) + scale.
fn check_grid_psnr(
    data: &[u8],
    codec: CodecChoice,
    gainmap: bool,
    threads: u32,
    min_yuv_psnr: f64,
    min_rgba_psnr: f64,
    name: &str,
) -> AvifResult<()> {
    let mut ref_decoder = decoder_for_data(data, codec, gainmap);
    ref_decoder.parse()?;
    ref_decoder.nth_image(0)?;
    let ref_full = decoded_image(&ref_decoder, gainmap).try_deep_clone()?;
    let base = base_clap_rect(&ref_full);
    let (bw, bh) = (base.width, base.height);

    let mut decoder = decoder_for_data(data, codec, gainmap);
    decoder.settings.max_threads = threads;
    decoder.parse()?;

    // 1. 1:1 identity scaled_size must be bit-exact with full decode (cropped to clap if any).
    let identity_opts = DecodeOptions {
        crop_rect: None,
        scaled_size: Some(size(bw, bh)),
    };
    decoder.nth_image_with_options(0, &identity_opts)?;
    let expected_identity = reference_crop_and_scale(&ref_full, None, Some(size(bw, bh)))?;
    let actual_identity = decoded_image(&decoder, gainmap);
    assert!(actual_identity.clap.is_none(), "{name}");
    assert_eq!(actual_identity.irot_angle, ref_full.irot_angle, "{name}");
    assert_eq!(actual_identity.imir_axis, ref_full.imir_axis, "{name}");
    assert!(
        are_images_equal(actual_identity, &expected_identity)?,
        "{name}: 1:1 scaled_size differs from unscaled decode"
    );

    // 2. Downscaled full-image (or clap) targets: verify high YUV & RGBA PSNR.
    for sz in [
        size((bw / 2).max(1), (bh / 2).max(1)),
        size((bw / 3).max(1), (bh / 3).max(1)),
        size((bw / 4).max(1), (bh / 4).max(1)),
        size(((bw * 3) / 5).max(1) | 1, ((bh * 2) / 5).max(1) | 1),
        size((bw / 2).max(1), bh),
        size(bw, (bh / 2).max(1)),
    ] {
        if sz.width < 16 || sz.height < 16 {
            continue;
        }
        let opts = DecodeOptions {
            crop_rect: None,
            scaled_size: Some(sz),
        };
        decoder
            .nth_image_with_options(0, &opts)
            .unwrap_or_else(|err| panic!("{name}: grid scale {sz:?} failed: {err:?}"));
        let actual = decoded_image(&decoder, gainmap);
        let expected = reference_crop_and_scale(&ref_full, None, Some(sz))?;
        assert_eq!(
            (actual.width, actual.height),
            (sz.width, sz.height),
            "{name}"
        );
        assert!(actual.clap.is_none(), "{name}");
        assert_eq!(actual.irot_angle, ref_full.irot_angle, "{name}");
        assert_eq!(actual.imir_axis, ref_full.imir_axis, "{name}");
        let yuv_p = psnr(actual, &expected)?;
        assert!(
            yuv_p >= min_yuv_psnr,
            "{name}: scale {sz:?} ({codec:?}, threads={threads}) \
             YUV PSNR {yuv_p:.2} < {min_yuv_psnr}"
        );
        let rgba_p = rgb_psnr(actual, &expected)?;
        assert!(
            rgba_p >= min_rgba_psnr,
            "{name}: scale {sz:?} ({codec:?}, threads={threads}) \
             RGBA PSNR {rgba_p:.2} < {min_rgba_psnr}"
        );
    }

    // 3. Combined crop_rect + scaled_size across cells.
    if bw >= 64 && bh >= 64 {
        let cross_crop = crop(
            (bw / 5) | 1,
            (bh / 5) | 1,
            ((bw * 3) / 5) | 1,
            ((bh * 3) / 5) | 1,
        );
        for sz in [
            size(cross_crop.width, cross_crop.height),
            size(
                (cross_crop.width / 2).max(1),
                (cross_crop.height / 2).max(1),
            ),
            size(
                ((cross_crop.width * 2) / 5).max(1),
                ((cross_crop.height * 3) / 7).max(1),
            ),
        ] {
            let opts = DecodeOptions {
                crop_rect: Some(cross_crop),
                scaled_size: Some(sz),
            };
            decoder.nth_image_with_options(0, &opts)?;
            let actual = decoded_image(&decoder, gainmap);
            let expected = reference_crop_and_scale(&ref_full, Some(cross_crop), Some(sz))?;
            assert_eq!(
                (actual.width, actual.height),
                (sz.width, sz.height),
                "{name}"
            );
            assert!(actual.clap.is_none(), "{name}");
            if sz == size(cross_crop.width, cross_crop.height) {
                assert!(
                    are_images_equal(actual, &expected)?,
                    "{name}: 1:1 crop+scale differs"
                );
            } else if sz.width >= 16 && sz.height >= 16 {
                let yuv_p = psnr(actual, &expected)?;
                assert!(
                    yuv_p >= min_yuv_psnr,
                    "{name}: crop {cross_crop:?} + scale {sz:?} \
                     YUV PSNR {yuv_p:.2} < {min_yuv_psnr}"
                );
                let rgba_p = rgb_psnr(actual, &expected)?;
                assert!(
                    rgba_p >= min_rgba_psnr,
                    "{name}: crop {cross_crop:?} + scale {sz:?} \
                     RGBA PSNR {rgba_p:.2} < {min_rgba_psnr}"
                );
            }
        }
    }

    // 4. Extreme downscales (2x2 and 1x1, where some grid cells map to 0 output pixels):
    // verify clean decode, exact output dimensions, and valid RGBA conversion.
    for extreme in [size(2.min(bw), 2.min(bh)), size(1, 1)] {
        decoder.nth_image_with_options(
            0,
            &DecodeOptions {
                crop_rect: None,
                scaled_size: Some(extreme),
            },
        )?;
        let actual = decoded_image(&decoder, gainmap);
        assert_eq!(
            (actual.width, actual.height),
            (extreme.width, extreme.height)
        );
        assert!(actual.clap.is_none());
        let rgba = to_rgba(actual)?;
        assert_eq!((rgba.width, rgba.height), (extreme.width, extreme.height));
    }

    // 5. Subsequent full decode restores original unscaled image.
    decoder.nth_image(0)?;
    assert!(
        are_images_equal(decoded_image(&decoder, gainmap), &ref_full)?,
        "{name}: full decode after grid scaled decodes differs"
    );
    Ok(())
}

// All 24 trimmed 3x3 262x262 grid files (400/420/422/444 x 8/10/12bpc x alpha).
#[test_matrix(
    [PixelFormat::Yuv400, PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444],
    [8, 10, 12],
    [false, true],
    [DAV1D, NATIVE]
)]
fn scale_decode_region_grids(
    yuv_format: PixelFormat,
    depth: u8,
    alpha: bool,
    codec: CodecChoice,
) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let yuv = yuv_format_name(yuv_format);
    let alpha_suffix = if alpha { "_alpha" } else { "" };
    let filename = format!("region/region_grid_{yuv}_{depth}bpc{alpha_suffix}.avif");
    let data = std::fs::read(get_test_file(&filename)).expect("Unable to read file");
    // The region_grid_* fixtures use ffmpeg's high-frequency `testsrc2` chart (1-pixel stripes
    // and text), where rounding non-divisor cell boundaries to integers / even chroma coordinates
    // shifts subpixel phase by <= 1 output pixel at seams (~24-31 dB YUV, ~22-30 dB RGBA), while
    // 1:1 and single-cell scales below are verified to be bit-exact.
    check_grid_psnr(&data, codec, false, 2, 23.5, 21.5, &filename)?;

    // Also check a single-cell crop_rect + scaled_size (strictly inside cell (0, 0), 88x88):
    // since only one cell contributes, cell-level scaling must be bit-exact with full + crop +
    // scale!
    let mut ref_decoder = decoder_for_data(&data, codec, false);
    ref_decoder.parse()?;
    ref_decoder.nth_image(0)?;
    let ref_full = ref_decoder.image().unwrap().try_deep_clone()?;

    let single_cell_crop = crop(6, 8, 60, 56);
    let single_cell_scale = size(27, 23);
    let mut decoder = decoder_for_data(&data, codec, false);
    decoder.parse()?;
    decoder.nth_image_with_options(
        0,
        &DecodeOptions {
            crop_rect: Some(single_cell_crop),
            scaled_size: Some(single_cell_scale),
        },
    )?;
    let expected =
        reference_crop_and_scale(&ref_full, Some(single_cell_crop), Some(single_cell_scale))?;
    assert!(
        are_images_equal(decoder.image().unwrap(), &expected)?,
        "{filename}: single-cell crop+scale must be bit-exact"
    );
    Ok(())
}

// Grid with clean aperture (240x240 3x3 grid, clap = (40, 50, 180, 160)), irot = 1, imir = 0.
#[test_matrix([DAV1D, NATIVE])]
fn scale_decode_grid_clean_aperture(codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let filename = "region/region_grid_clap_irot_imir.avif";
    let data = std::fs::read(get_test_file(filename)).expect("Unable to read file");
    check_grid_psnr(&data, codec, false, 2, 23.5, 19.5, filename)?;

    // Across all 9 grid cells, a 2:1 box-filter downscale whose per-cell widths [40, 80, 40] ->
    // [20, 40, 20] and heights [20, 80, 20] -> [10, 40, 10] are exact even integers has zero
    // boundary rounding and disjoint 2x2 box kernels, so cell-level scaling is bit-exact.
    let mut ref_dec = decoder_for_data(&data, codec, false);
    ref_dec.parse()?;
    ref_dec.nth_image(0)?;
    let ref_full = ref_dec.image().unwrap().try_deep_clone()?;
    let exact_crop = crop(0, 10, 160, 120);
    let exact_scale = size(80, 60);
    let mut exact_dec = decoder_for_data(&data, codec, false);
    exact_dec.parse()?;
    exact_dec.nth_image_with_options(
        0,
        &DecodeOptions {
            crop_rect: Some(exact_crop),
            scaled_size: Some(exact_scale),
        },
    )?;
    let expected_exact = reference_crop_and_scale(&ref_full, Some(exact_crop), Some(exact_scale))?;
    assert!(
        are_images_equal(exact_dec.image().unwrap(), &expected_exact)?,
        "exact 2:1 9-cell crop+scale must be bit-exact"
    );

    let mut decoder = decoder_for_data(&data, codec, false);
    decoder.parse()?;
    assert_eq!(
        (
            decoder.image().unwrap().width,
            decoder.image().unwrap().height
        ),
        (240, 240)
    );
    assert!(decoder.image().unwrap().clap.is_some());
    assert_eq!(decoder.image().unwrap().irot_angle, Some(1));
    assert_eq!(decoder.image().unwrap().imir_axis, Some(0));

    // Upscaling relative to clap (181 > 180, even though 181 <= 240) must fail.
    assert_eq!(
        decoder.nth_image_with_options(
            0,
            &DecodeOptions {
                crop_rect: None,
                scaled_size: Some(size(181, 160)),
            },
        ),
        Err(AvifError::InvalidArgument)
    );
    // Upscaling relative to user crop_rect must also fail.
    assert_eq!(
        decoder.nth_image_with_options(
            0,
            &DecodeOptions {
                crop_rect: Some(crop(10, 10, 80, 70)),
                scaled_size: Some(size(80, 71)),
            },
        ),
        Err(AvifError::InvalidArgument)
    );

    // Invalid clap with StrictnessFlag::ClapValid fails when scaling; with Strictness::None,
    // invalid clap is ignored and scaling uses the full 240x240 grid dimensions.
    let mut patched = data.clone();
    let clap_pos = patched
        .windows(4)
        .position(|w| w == b"clap")
        .expect("clap box not found");
    // Patch cleanApertureWidthD (at offset +8 inside clap payload, i.e. clap_pos + 8) to 0.
    patched[clap_pos + 8..clap_pos + 12].copy_from_slice(&0u32.to_be_bytes());

    let mut strict_dec = decoder_for_data(&patched, codec, false);
    strict_dec.parse()?;
    assert!(strict_dec
        .nth_image_with_options(
            0,
            &DecodeOptions {
                crop_rect: None,
                scaled_size: Some(size(120, 120)),
            },
        )
        .is_err());

    let mut loose_dec = decoder_for_data(&patched, codec, false);
    loose_dec.settings.strictness = decoder::Strictness::None;
    loose_dec.parse()?;
    loose_dec.nth_image_with_options(
        0,
        &DecodeOptions {
            crop_rect: None,
            scaled_size: Some(size(200, 200)), // > 180, valid because invalid clap is ignored!
        },
    )?;
    assert_eq!(
        (
            loose_dec.image().unwrap().width,
            loose_dec.image().unwrap().height
        ),
        (200, 200)
    );
    Ok(())
}

// Mixed grid / non-grid combinations, shared grid tiles, thumbnails, and multithreading.
#[test_matrix([1, 4], [DAV1D, NATIVE])]
fn scale_decode_mixed_grids_and_thumbnails(threads: u32, codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    for (filename, gainmap, min_yuv, min_rgba) in [
        ("sofa_grid1x5_420.avif", false, 28.0, 27.0),
        ("sofa_grid1x5_420_random_dimg_order.avif", false, 25.0, 24.0),
        ("color_grid_alpha_nogrid.avif", false, 25.0, 24.0),
        (
            "color_grid_alpha_grid_gainmap_nogrid.avif",
            false,
            25.0,
            24.0,
        ),
        (
            "color_nogrid_alpha_nogrid_gainmap_grid.avif",
            true,
            21.0,
            20.0,
        ),
        ("color_grid_gainmap_different_grid.avif", false, 25.0, 24.0),
        ("color_grid_gainmap_different_grid.avif", true, 21.0, 20.0),
        ("color_grid_alpha_grid_shared_tiles.avif", false, 25.0, 24.0),
        ("grid_icc_individual_cells.avif", false, 25.0, 24.0),
        ("grid_nclx_individual_cells.avif", false, 25.0, 24.0),
    ] {
        let data = std::fs::read(get_test_file(filename)).expect("Unable to read file");
        check_grid_psnr(&data, codec, gainmap, threads, min_yuv, min_rgba, filename)?;
    }
    // Non-grid items inside mixed grid/non-grid files must be bit-exact.
    for (filename, gainmap) in [
        ("color_grid_alpha_grid_gainmap_nogrid.avif", true),
        ("color_nogrid_alpha_nogrid_gainmap_grid.avif", false),
    ] {
        let data = std::fs::read(get_test_file(filename)).expect("Unable to read file");
        check_non_grid_exact(&data, codec, gainmap, false, false, 0, filename)?;
    }

    // Thumbnails: thumb 0 is a 2x1 grid with per-tile auxl alpha; thumb 1 is a standalone item.
    let thumb_data = std::fs::read(get_test_file("thumbnails_grid_per_tile_alpha.avif"))
        .expect("Unable to read thumbnails");
    let mut thumb_ref = decoder_for_data(&thumb_data, codec, false);
    thumb_ref.settings.source = decoder::Source::Thumbnail;
    thumb_ref.parse()?;
    assert_eq!(thumb_ref.image_count(), 2);

    let mut thumb_dec = decoder_for_data(&thumb_data, codec, false);
    thumb_dec.settings.source = decoder::Source::Thumbnail;
    thumb_dec.settings.max_threads = threads;
    thumb_dec.parse()?;

    // Thumbnail 0 (256x200 grid with per-tile alpha).
    thumb_ref.nth_image(0)?;
    let ref_t0 = thumb_ref.image().unwrap().try_deep_clone()?;
    let opts_t0 = DecodeOptions {
        crop_rect: Some(crop(30, 20, 190, 150)),
        scaled_size: Some(size(95, 75)),
    };
    thumb_dec.nth_image_with_options(0, &opts_t0)?;
    let exp_t0 = reference_crop_and_scale(&ref_t0, opts_t0.crop_rect, opts_t0.scaled_size)?;
    assert!(psnr(thumb_dec.image().unwrap(), &exp_t0)? >= 28.0);

    // Thumbnail 1 (64x80 non-grid without alpha): bit-exact.
    thumb_ref.nth_image(1)?;
    let ref_t1 = thumb_ref.image().unwrap().try_deep_clone()?;
    let opts_t1 = DecodeOptions {
        crop_rect: None,
        scaled_size: Some(size(32, 40)),
    };
    thumb_dec.nth_image_with_options(1, &opts_t1)?;
    let exp_t1 = reference_crop_and_scale(&ref_t1, opts_t1.crop_rect, opts_t1.scaled_size)?;
    assert!(are_images_equal(thumb_dec.image().unwrap(), &exp_t1)?);
    Ok(())
}

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

// Selective IO on a 1x5 grid (1024x770, cells 1024x154): when crop_rect + scaled_size
// intersects only middle cells, unused cells (0 and 4) are never read and may even contain
// corrupted mdat payloads without breaking decoding.
#[test_matrix([DAV1D, NATIVE])]
fn scale_decode_grid_selective_io_and_corrupted_unused_cells(codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let data = std::fs::read(get_test_file("sofa_grid1x5_420.avif")).expect("Unable to read file");

    let reads = Rc::new(RefCell::new(Vec::new()));
    let mut probe = decoder::Decoder::default();
    probe.settings.codec_choice = codec;
    probe.set_io(Box::new(TrackingIO {
        data: data.clone(),
        reads: reads.clone(),
    }));
    probe.parse()?;

    // Record byte ranges of cell 0 (y in 0..154) and cell 4 (y in 616..770).
    reads.borrow_mut().clear();
    probe.nth_image_with_options(
        0,
        &DecodeOptions {
            crop_rect: Some(crop(0, 0, 64, 64)),
            scaled_size: Some(size(32, 32)),
        },
    )?;
    let cell0_reads = reads.borrow().clone();
    assert!(!cell0_reads.is_empty());

    reads.borrow_mut().clear();
    probe.nth_image_with_options(
        0,
        &DecodeOptions {
            crop_rect: Some(crop(0, 700, 64, 64)),
            scaled_size: Some(size(32, 32)),
        },
    )?;
    let cell4_reads = reads.borrow().clone();
    assert!(!cell4_reads.is_empty());

    // Now decode a middle crop (y = 160..410, intersecting only cells 1 and 2) + scale.
    let mid_opts = DecodeOptions {
        crop_rect: Some(crop(100, 160, 500, 250)),
        scaled_size: Some(size(200, 100)),
    };
    reads.borrow_mut().clear();
    probe.nth_image_with_options(0, &mid_opts)?;
    let mid_reads = reads.borrow().clone();
    assert!(!mid_reads.is_empty());
    let expected_mid = probe.image().unwrap().try_deep_clone()?;

    let overlaps = |(a0, a1): (usize, usize), (b0, b1): (usize, usize)| a0 < b1 && b0 < a1;
    for &r in &mid_reads {
        for &c0 in &cell0_reads {
            assert!(
                !overlaps(r, c0),
                "middle crop+scale read cell 0 bytes {c0:?}"
            );
        }
        for &c4 in &cell4_reads {
            assert!(
                !overlaps(r, c4),
                "middle crop+scale read cell 4 bytes {c4:?}"
            );
        }
    }

    // Corrupt cells 0 and 4 with 0xFF and verify full decode fails while mid_opts succeeds!
    let mut corrupted = data;
    for &(s, e) in cell0_reads.iter().chain(cell4_reads.iter()) {
        corrupted[s..e].fill(0xFF);
    }
    let mut dec = decoder_for_data(&corrupted, codec, false);
    dec.parse()?;
    assert!(dec.nth_image(0).is_err());
    assert!(dec
        .nth_image_with_options(
            0,
            &DecodeOptions {
                crop_rect: None,
                scaled_size: Some(size(256, 192)),
            }
        )
        .is_err());

    dec.nth_image_with_options(0, &mid_opts)?;
    assert!(are_images_equal(dec.image().unwrap(), &expected_mid)?);
    Ok(())
}

// Cache hits, state transitions across full / cropped / scaled / cropped+scaled / failed decodes,
// and nth_image_properties invariance.
#[test_matrix(
    [
        ("alpha.avif", false),
        ("seine_sdr_gainmap_srgb.avif", true),
        ("sofa_grid1x5_420.avif", false),
        ("color_grid_alpha_nogrid.avif", false),
        ("region/region_grid_clap_irot_imir.avif", false),
    ],
    [DAV1D, NATIVE]
)]
fn scale_decode_cache_and_state_transitions(
    file_and_gainmap: (&str, bool),
    codec: CodecChoice,
) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let (filename, gainmap) = file_and_gainmap;
    let data = std::fs::read(get_test_file(filename)).expect("Unable to read file");

    let mut ref_decoder = decoder_for_data(&data, codec, gainmap);
    ref_decoder.parse()?;
    let orig_props = ref_decoder.nth_image_properties(0)?;
    ref_decoder.nth_image(0)?;
    let ref_full = decoded_image(&ref_decoder, gainmap).try_deep_clone()?;
    let base = base_clap_rect(&ref_full);
    let (bw, bh) = (base.width, base.height);

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

    let scale1 = DecodeOptions {
        crop_rect: None,
        scaled_size: Some(size(bw / 2, bh / 2)),
    };
    let scale2 = DecodeOptions {
        crop_rect: None,
        scaled_size: Some(size(bw / 3, bh / 3)),
    };
    let crop_only = DecodeOptions {
        crop_rect: Some(crop(4, 6, bw / 2, bh / 2)),
        scaled_size: None,
    };
    let crop_and_scale = DecodeOptions {
        crop_rect: Some(crop(4, 6, bw / 2, bh / 2)),
        scaled_size: Some(size(bw / 4, bh / 4)),
    };

    // 1. Scale decode, then cache hit with identical options (zero reads).
    decoder.nth_image_with_options(0, &scale1)?;
    let img_scale1 = decoded_image(&decoder, gainmap).try_deep_clone()?;
    assert_eq!((img_scale1.width, img_scale1.height), (bw / 2, bh / 2));
    reads.borrow_mut().clear();
    decoder.nth_image_with_options(0, &scale1)?;
    assert!(reads.borrow().is_empty(), "{filename}: expected cache hit");
    assert!(are_images_equal(
        decoded_image(&decoder, gainmap),
        &img_scale1
    )?);

    // 2. nth_image_properties still reports original unscaled dimensions & clap.
    let props = decoder.nth_image_properties(0)?;
    assert_eq!(
        (props.width, props.height),
        (orig_props.width, orig_props.height)
    );
    assert_eq!(props.clap, orig_props.clap);
    assert_eq!(props.depth, orig_props.depth);
    assert_eq!(props.yuv_format, orig_props.yuv_format);

    // 3. Different scaled_size triggers a new decode.
    decoder.nth_image_with_options(0, &scale2)?;
    assert!(!reads.borrow().is_empty());
    assert_eq!(
        (
            decoded_image(&decoder, gainmap).width,
            decoded_image(&decoder, gainmap).height
        ),
        (bw / 3, bh / 3)
    );

    // 4. Transition to crop_only, then crop_and_scale, then cache hit on crop_and_scale.
    decoder.nth_image_with_options(0, &crop_only)?;
    assert_eq!(
        (
            decoded_image(&decoder, gainmap).width,
            decoded_image(&decoder, gainmap).height
        ),
        (bw / 2, bh / 2)
    );
    decoder.nth_image_with_options(0, &crop_and_scale)?;
    let img_crop_scale = decoded_image(&decoder, gainmap).try_deep_clone()?;
    assert_eq!(
        (img_crop_scale.width, img_crop_scale.height),
        (bw / 4, bh / 4)
    );
    reads.borrow_mut().clear();
    decoder.nth_image_with_options(0, &crop_and_scale)?;
    assert!(reads.borrow().is_empty());
    assert!(are_images_equal(
        decoded_image(&decoder, gainmap),
        &img_crop_scale
    )?);

    // 5. next_image past the end fails with NoImagesRemaining and leaves the scaled image intact.
    assert_eq!(decoder.next_image(), Err(AvifError::NoImagesRemaining));
    assert!(are_images_equal(
        decoded_image(&decoder, gainmap),
        &img_crop_scale
    )?);

    // 6. Invalid scaled_size fails before touching the current decoded image.
    assert_eq!(
        decoder.nth_image_with_options(
            0,
            &DecodeOptions {
                crop_rect: None,
                scaled_size: Some(size(bw + 1, bh)),
            }
        ),
        Err(AvifError::InvalidArgument)
    );
    assert!(are_images_equal(
        decoded_image(&decoder, gainmap),
        &img_crop_scale
    )?);

    // 7. Full decode restores original image, followed by another scaled decode.
    decoder.nth_image(0)?;
    assert!(are_images_equal(
        decoded_image(&decoder, gainmap),
        &ref_full
    )?);
    decoder.nth_image_with_options(0, &scale1)?;
    assert!(are_images_equal(
        decoded_image(&decoder, gainmap),
        &img_scale1
    )?);
    Ok(())
}

// Validation and unsupported-mode error cases.
#[test_matrix([DAV1D, NATIVE])]
fn scale_decode_errors(codec: CodecChoice) -> AvifResult<()> {
    if !HAS_NON_ANDROID_DECODER
        || (!cfg!(feature = "dav1d") && codec == DAV1D)
        || (!cfg!(feature = "native_av1") && codec == NATIVE)
    {
        return Ok(());
    }
    let data = std::fs::read(get_test_file("sofa_grid1x5_420.avif")).expect("Unable to read file");

    if !cfg!(feature = "libyuv") {
        let mut no_yuv = decoder_for_data(&data, codec, false);
        no_yuv.parse()?;
        assert_eq!(
            no_yuv.nth_image_with_options(
                0,
                &DecodeOptions {
                    crop_rect: None,
                    scaled_size: Some(size(64, 64)),
                }
            ),
            Err(AvifError::NotImplemented)
        );
        return Ok(());
    }

    let mut decoder = decoder_for_data(&data, codec, false);
    decoder.parse()?;
    let (w, h) = (
        decoder.image().unwrap().width,
        decoder.image().unwrap().height,
    );

    // Zero width or height, or upscaling.
    for bad_opts in [
        DecodeOptions {
            crop_rect: None,
            scaled_size: Some(size(0, 10)),
        },
        DecodeOptions {
            crop_rect: None,
            scaled_size: Some(size(10, 0)),
        },
        DecodeOptions {
            crop_rect: None,
            scaled_size: Some(size(0, 0)),
        },
        DecodeOptions {
            crop_rect: None,
            scaled_size: Some(size(w + 1, h)),
        },
        DecodeOptions {
            crop_rect: None,
            scaled_size: Some(size(w, h + 1)),
        },
        DecodeOptions {
            crop_rect: Some(crop(10, 10, 100, 80)),
            scaled_size: Some(size(101, 80)),
        },
        DecodeOptions {
            crop_rect: Some(crop(10, 10, 100, 80)),
            scaled_size: Some(size(100, 81)),
        },
    ] {
        assert_eq!(
            decoder.nth_image_with_options(0, &bad_opts),
            Err(AvifError::InvalidArgument),
            "{bad_opts:?}"
        );
        assert_eq!(
            decoder.next_image_with_options(&bad_opts),
            Err(AvifError::InvalidArgument),
            "{bad_opts:?}"
        );
    }

    // Progressive / incremental + scaled_size -> NotImplemented.
    let valid_scale = DecodeOptions {
        crop_rect: None,
        scaled_size: Some(size(128, 96)),
    };
    let mut prog_dec = decoder_for_data(&data, codec, false);
    prog_dec.settings.allow_progressive = true;
    prog_dec.parse()?;
    assert_eq!(
        prog_dec.next_image_with_options(&valid_scale),
        Err(AvifError::NotImplemented)
    );

    let mut inc_dec = decoder_for_data(&data, codec, false);
    inc_dec.settings.allow_incremental = true;
    inc_dec.parse()?;
    assert_eq!(
        inc_dec.next_image_with_options(&valid_scale),
        Err(AvifError::NotImplemented)
    );

    // GainMap content without a gain map -> NoContent.
    let mut no_gm = decoder_for_data(&data, codec, true);
    no_gm.parse()?;
    assert_eq!(
        no_gm.nth_image_with_options(0, &valid_scale),
        Err(AvifError::NoContent)
    );

    // Color + GainMap simultaneously when gainmap is present -> NotImplemented;
    // when gainmap is NOT present, ImageContentType::All succeeds!
    let mut all_no_gm = decoder_for_data(&data, codec, false);
    all_no_gm.settings.image_content_to_decode = ImageContentType::All;
    all_no_gm.parse()?;
    all_no_gm.nth_image_with_options(0, &valid_scale)?;
    assert_eq!(
        (
            all_no_gm.image().unwrap().width,
            all_no_gm.image().unwrap().height
        ),
        (128, 96)
    );

    let gm_data =
        std::fs::read(get_test_file("seine_sdr_gainmap_srgb.avif")).expect("Unable to read file");
    for content in [ImageContentType::All, ImageContentType::ColorAndGainMap] {
        let mut both_dec = decoder_for_data(&gm_data, codec, false);
        both_dec.settings.image_content_to_decode = content;
        both_dec.parse()?;
        assert_eq!(
            both_dec.nth_image_with_options(0, &valid_scale),
            Err(AvifError::NotImplemented)
        );
    }
    Ok(())
}
