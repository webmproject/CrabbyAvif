// Copyright 2025 Google LLC
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

use crabby_avif::image::*;
use crabby_avif::reformat::rgb;
use crabby_avif::reformat::rgb::ChromaDownsampling;
use crabby_avif::*;

use test_case::test_matrix;

#[derive(Default)]
struct RgbToYuvParam {
    rgb_depth: u8,
    yuv_depth: u8,
    rgb_format: rgb::Format,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    matrix_coefficients: MatrixCoefficients,
    #[allow(unused)]
    chroma_downsampling: ChromaDownsampling,
    add_noise: bool,
    rgb_step: u32,
    max_average_abs_diff: f64,
    min_psnr: f64,
}

fn fill_rgb_image_channel(
    rgb: &mut rgb::Image,
    channel_offset: usize,
    value: u16,
) -> AvifResult<()> {
    let channel_count = rgb.channel_count() as usize;
    let pixel_width = channel_count * rgb.width as usize;
    assert!(channel_offset < channel_count);
    for y in 0..rgb.height {
        if rgb.depth == 8 {
            let row = &mut rgb.row_mut(y)?[..pixel_width];
            for pixels in row.chunks_exact_mut(channel_count) {
                pixels[channel_offset] = value as u8;
            }
        } else {
            let row = &mut rgb.row16_mut(y)?[..pixel_width];
            for pixels in row.chunks_exact_mut(channel_count) {
                pixels[channel_offset] = value;
            }
        }
    }
    Ok(())
}

fn compute_diff_sum(
    rgb1: &rgb::Image,
    rgb2: &rgb::Image,
    abs_diff_sum: &mut i64,
    sq_diff_sum: &mut i64,
    max_abs_diff: &mut i64,
) -> AvifResult<()> {
    assert_eq!(rgb1.depth, rgb2.depth);
    assert!(rgb1.format == rgb2.format);
    let pixel_width = (rgb1.width * rgb1.channel_count()) as usize;
    for y in 0..rgb1.height {
        if rgb1.depth == 8 {
            let row1 = &rgb1.row(y)?[..pixel_width];
            let row2 = &rgb2.row(y)?[..pixel_width];
            for x in 0..pixel_width {
                let diff = row2[x] as i64 - row1[x] as i64;
                *abs_diff_sum += diff.abs();
                *sq_diff_sum += diff * diff;
                *max_abs_diff = std::cmp::max(*max_abs_diff, diff.abs());
            }
        } else {
            let row1 = &rgb1.row16(y)?[..pixel_width];
            let row2 = &rgb2.row16(y)?[..pixel_width];
            for x in 0..pixel_width {
                let diff = row2[x] as i64 - row1[x] as i64;
                *abs_diff_sum += diff.abs();
                *sq_diff_sum += diff * diff;
                *max_abs_diff = std::cmp::max(*max_abs_diff, diff.abs());
            }
        }
    }
    Ok(())
}

fn psnr(sq_diff_sum: f64, num_diffs: f64, max_abs_diff: f64) -> f64 {
    if sq_diff_sum == 0.0 {
        return 99.0;
    }
    let distortion = sq_diff_sum / (num_diffs * max_abs_diff * max_abs_diff);
    if distortion > 0.0 {
        (-10.0 * distortion.log10()).min(98.9)
    } else {
        98.9
    }
}

fn rgb_to_yuv_whole_range(p: &RgbToYuvParam) -> AvifResult<()> {
    let width = 4;
    let height = 4;
    let mut image = image::Image {
        width,
        height,
        depth: p.yuv_depth,
        yuv_format: p.yuv_format,
        yuv_range: p.yuv_range,
        matrix_coefficients: p.matrix_coefficients,
        ..Default::default()
    };
    image.allocate_planes(Category::Color)?;
    if p.rgb_format.has_alpha() {
        image.allocate_planes(Category::Alpha)?;
    }
    let mut src_rgb = rgb::Image {
        width,
        height,
        depth: p.rgb_depth,
        format: p.rgb_format,
        ..Default::default()
    };
    src_rgb.allocate()?;
    let mut dst_rgb = rgb::Image {
        width,
        height,
        depth: p.rgb_depth,
        format: p.rgb_format,
        ..Default::default()
    };
    dst_rgb.allocate()?;
    let rgb_max_channel = src_rgb.max_channel();
    if p.rgb_format.has_alpha() {
        fill_rgb_image_channel(&mut src_rgb, p.rgb_format.alpha_offset(), rgb_max_channel)?;
    }
    let mut abs_diff_sum = 0i64;
    let mut sq_diff_sum = 0i64;
    let mut max_abs_diff = 0i64;
    let mut num_diffs = 0i64;
    let max_value = (rgb_max_channel - if p.add_noise { 15 } else { 0 }) as u32;
    let rgb_step = p.rgb_step;
    for r in (0..max_value + rgb_step).step_by(rgb_step as usize) {
        let value = std::cmp::min(r, max_value) as u16;
        fill_rgb_image_channel(&mut src_rgb, p.rgb_format.r_offset(), value)?;
        if p.add_noise {
            todo!();
        }
        if p.yuv_format == PixelFormat::Yuv400 {
            todo!();
        } else {
            for g in (0..max_value + rgb_step).step_by(rgb_step as usize) {
                let value = std::cmp::min(g, max_value) as u16;
                fill_rgb_image_channel(&mut src_rgb, p.rgb_format.g_offset(), value)?;
                if p.add_noise {
                    todo!();
                }
                for b in (0..max_value + rgb_step).step_by(rgb_step as usize) {
                    let value = std::cmp::min(b, max_value) as u16;
                    fill_rgb_image_channel(&mut src_rgb, p.rgb_format.b_offset(), value)?;
                    if p.add_noise {
                        todo!();
                    }
                    src_rgb.convert_to_yuv(&mut image)?;
                    dst_rgb.convert_from_yuv(&image)?;
                    compute_diff_sum(
                        &src_rgb,
                        &dst_rgb,
                        &mut abs_diff_sum,
                        &mut sq_diff_sum,
                        &mut max_abs_diff,
                    )?;
                    num_diffs += (src_rgb.width * src_rgb.height * 3) as i64;
                }
            }
        }
    }
    let average_abs_diff = abs_diff_sum as f64 / num_diffs as f64;
    let psnr = psnr(sq_diff_sum as f64, num_diffs as f64, rgb_max_channel as f64);
    assert!(average_abs_diff <= p.max_average_abs_diff);
    assert!(psnr >= p.min_psnr);
    Ok(())
}

#[test_matrix(
    [8, 10, 12, 16],
    [8, 10, 12, 16],
    [
        rgb::Format::Rgb, rgb::Format::Rgba, rgb::Format::Argb,
        rgb::Format::Bgr, rgb::Format::Bgra, rgb::Format::Abgr,
    ],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444],
    [YuvRange::Full, YuvRange::Limited],
    [
        ChromaDownsampling::Automatic,
        ChromaDownsampling::Fastest,
        ChromaDownsampling::BestQuality,
        ChromaDownsampling::Average,
    ],
    [MatrixCoefficients::Bt601]
)]
fn exhaustive_settings(
    rgb_depth: u8,
    yuv_depth: u8,
    rgb_format: rgb::Format,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    chroma_downsampling: ChromaDownsampling,
    matrix_coefficients: MatrixCoefficients,
) -> AvifResult<()> {
    rgb_to_yuv_whole_range(&RgbToYuvParam {
        rgb_depth,
        yuv_depth,
        rgb_format,
        yuv_format,
        yuv_range,
        matrix_coefficients,
        chroma_downsampling,
        add_noise: false,
        // Only try the minimum and maximum values.
        rgb_step: (1 << rgb_depth) - 1,
        // Barely check the results, just for coverage.
        max_average_abs_diff: ((1 << rgb_depth) - 1) as f64,
        min_psnr: 5.0,
    })
}

#[test_matrix(
    [8, 10, 12, 16],
    [8, 10, 12, 16],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444],
    [YuvRange::Full, YuvRange::Limited],
    [ChromaDownsampling::Fastest, ChromaDownsampling::Automatic],
    [
        MatrixCoefficients::Bt709,
        MatrixCoefficients::Unspecified,
        MatrixCoefficients::Fcc,
        MatrixCoefficients::Bt470bg,
        MatrixCoefficients::Bt601,
        MatrixCoefficients::Smpte240,
        MatrixCoefficients::Bt2020Ncl,
        MatrixCoefficients::ChromaDerivedNcl,
    ]
)]
fn all_matrix_coefficients(
    rgb_depth: u8,
    yuv_depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    chroma_downsampling: ChromaDownsampling,
    matrix_coefficients: MatrixCoefficients,
) -> AvifResult<()> {
    rgb_to_yuv_whole_range(&RgbToYuvParam {
        rgb_depth,
        yuv_depth,
        rgb_format: rgb::Format::Rgba,
        yuv_format,
        yuv_range,
        matrix_coefficients,
        chroma_downsampling,
        add_noise: false,
        // Only try the minimum and maximum values.
        rgb_step: (1 << rgb_depth) - 1,
        // Barely check the results, just for coverage.
        max_average_abs_diff: ((1 << rgb_depth) - 1) as f64,
        min_psnr: 5.0,
    })
}

#[test]
fn default_8bit_png_to_avif() -> AvifResult<()> {
    rgb_to_yuv_whole_range(&RgbToYuvParam {
        rgb_depth: 8,
        yuv_depth: 8,
        rgb_format: rgb::Format::Rgba,
        yuv_format: PixelFormat::Yuv420,
        yuv_range: YuvRange::Full,
        matrix_coefficients: MatrixCoefficients::Bt601,
        chroma_downsampling: ChromaDownsampling::Automatic,
        add_noise: false,
        rgb_step: 3,
        max_average_abs_diff: 2.88,
        min_psnr: 36.0,
    })
}

#[test_matrix([PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444])]
fn any_subsampling_8bit(yuv_format: PixelFormat) -> AvifResult<()> {
    rgb_to_yuv_whole_range(&RgbToYuvParam {
        rgb_depth: 8,
        yuv_depth: 8,
        rgb_format: rgb::Format::Rgba,
        yuv_format,
        yuv_range: YuvRange::Full,
        matrix_coefficients: MatrixCoefficients::Bt601,
        chroma_downsampling: ChromaDownsampling::Automatic,
        add_noise: false,
        rgb_step: 17,
        max_average_abs_diff: 0.84,
        min_psnr: 45.0,
    })
}

#[test_matrix(
    [rgb::Format::Rgba, rgb::Format::Bgr],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444],
    [(8, 61, 2.96, 36.0), (10, 211, 2.83, 47.0), (12, 809, 2.82, 52.0), (16, 16001, 2.82, 80.0)]
)]
fn all_same_bitdepths(
    rgb_format: rgb::Format,
    yuv_format: PixelFormat,
    params: (u8, u32, f64, f64),
) -> AvifResult<()> {
    rgb_to_yuv_whole_range(&RgbToYuvParam {
        rgb_depth: params.0,
        yuv_depth: params.0,
        rgb_format,
        yuv_format,
        yuv_range: YuvRange::Limited,
        matrix_coefficients: MatrixCoefficients::Bt601,
        chroma_downsampling: ChromaDownsampling::Automatic,
        add_noise: false,
        rgb_step: params.1,
        max_average_abs_diff: params.2,
        min_psnr: params.3,
    })
}
