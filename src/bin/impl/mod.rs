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

#![allow(clippy::disallowed_methods)]
#![allow(clippy::disallowed_macros)]

#[cfg(test)]
mod crabbyavif_test;
mod gainmap;

use clap::value_parser;
use clap::Parser;

use crabby_avif::decoder::*;
#[cfg(feature = "encoder")]
use crabby_avif::encoder::*;
use crabby_avif::utils::clap::CleanAperture;
use crabby_avif::utils::clap::CropRect;
use crabby_avif::utils::IFraction;
use crabby_avif::utils::UFraction;
use crabby_avif::*;

#[cfg(feature = "encoder")]
use crabby_avif::utils::reader::Config;
#[cfg(feature = "encoder")]
use crabby_avif::utils::reader::Reader;

#[cfg(feature = "jpeg")]
use crabby_avif::utils::writer::jpeg::JpegWriter;
#[cfg(feature = "png")]
use crabby_avif::utils::writer::png::PngWriter;
use crabby_avif::utils::writer::y4m::Y4MWriter;
use crabby_avif::utils::writer::Writer;

use std::fs::File;
#[cfg(feature = "encoder")]
use std::io;
#[cfg(feature = "encoder")]
use std::io::Read;
#[cfg(feature = "encoder")]
use std::io::Write;
use std::num::NonZero;
use std::path::Path;

fn depth_parser(s: &str) -> Result<u8, String> {
    match s.parse::<u8>() {
        Ok(8) => Ok(8),
        Ok(10) => Ok(10),
        Ok(12) => Ok(12),
        Ok(16) => Ok(16),
        _ => Err("Value must be one of 8, 10, 12 or 16".into()),
    }
}

fn repetition_count_parser(s: &str) -> Result<RepetitionCount, String> {
    if s == "infinite" {
        Ok(RepetitionCount::Infinite)
    } else {
        match s.parse::<i32>() {
            Ok(count) => Ok(RepetitionCount::create_from(count)),
            _ => Err(format!("Invalid value for repetition count: {s}")),
        }
    }
}

macro_rules! split_and_check_count {
    ($parameter: literal, $input:ident, $delimiter:literal, $count:literal, $type:ty) => {{
        let values: Result<Vec<_>, _> = $input
            .split($delimiter)
            .map(|x| x.parse::<$type>())
            .collect();
        if values.is_err() {
            return Err(format!("Invalid {} string", $parameter));
        }
        let values = values.unwrap();
        if values.len() != $count {
            return Err(format!(
                "Invalid {} string. Expecting exactly {} values separated with a \"{}\"",
                $parameter, $count, $delimiter
            ));
        }
        values
    }};
}

fn clap_parser(s: &str) -> Result<CleanAperture, String> {
    let values = split_and_check_count!("clap", s, ",", 8, i32);
    let values: Vec<_> = values.into_iter().map(|x| x as u32).collect();
    Ok(CleanAperture {
        width: UFraction(
            *values.first().ok_or("Invalid clap string")?,
            *values.get(1).ok_or("Invalid clap string")?,
        ),
        height: UFraction(
            *values.get(2).ok_or("Invalid clap string")?,
            *values.get(3).ok_or("Invalid clap string")?,
        ),
        horiz_off: UFraction(
            *values.get(4).ok_or("Invalid clap string")?,
            *values.get(5).ok_or("Invalid clap string")?,
        ),
        vert_off: UFraction(
            *values.get(6).ok_or("Invalid clap string")?,
            *values.get(7).ok_or("Invalid clap string")?,
        ),
    })
}

fn crop_parser(s: &str) -> Result<CropRect, String> {
    let values = split_and_check_count!("crop", s, ",", 4, u32);
    Ok(CropRect {
        x: *values.first().ok_or("Invalid crop string")?,
        y: *values.get(1).ok_or("Invalid crop string")?,
        width: *values.get(2).ok_or("Invalid crop string")?,
        height: *values.get(3).ok_or("Invalid crop string")?,
    })
}

fn target_size_parser(s: &str) -> Result<ImageSize, String> {
    let values = split_and_check_count!("target-size", s, ",", 2, u32);
    Ok(ImageSize {
        width: *values.first().ok_or("Invalid target-size string")?,
        height: *values.get(1).ok_or("Invalid target-size string")?,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GridSize {
    cols: u32,
    rows: u32,
}

fn grid_parser(s: &str) -> Result<GridSize, String> {
    let values = split_and_check_count!("grid", s, "x", 2, u32);
    let cols = *values.first().ok_or("Invalid grid string")?;
    let rows = *values.get(1).ok_or("Invalid grid string")?;
    if cols == 0 || cols > 256 || rows == 0 || rows > 256 || cols * rows > 65535 {
        return Err(format!("Invalid grid dimensions: {s}"));
    }
    Ok(GridSize { cols, rows })
}

fn clli_parser(s: &str) -> Result<ContentLightLevelInformation, String> {
    let values = split_and_check_count!("clli", s, ",", 2, u16);
    Ok(ContentLightLevelInformation {
        max_cll: *values.first().ok_or("Invalid clli string")?,
        max_pall: *values.get(1).ok_or("Invalid clli string")?,
    })
}

fn pasp_parser(s: &str) -> Result<PixelAspectRatio, String> {
    let values = split_and_check_count!("pasp", s, ",", 2, u32);
    Ok(PixelAspectRatio {
        h_spacing: *values.first().ok_or("Invalid pasp string")?,
        v_spacing: *values.get(1).ok_or("Invalid pasp string")?,
    })
}

fn cicp_parser(s: &str) -> Result<Nclx, String> {
    let values = split_and_check_count!("cicp", s, "/", 3, u16);
    Ok(Nclx {
        color_primaries: (*values.first().ok_or("Invalid cicp string")?).into(),
        transfer_characteristics: (*values.get(1).ok_or("Invalid cicp string")?).into(),
        matrix_coefficients: (*values.get(2).ok_or("Invalid cicp string")?).into(),
        ..Default::default()
    })
}

fn scaling_mode_parser(s: &str) -> Result<IFraction, String> {
    let values = split_and_check_count!("scaling_mode", s, "/", 2, i32);
    Ok(IFraction(
        *values.first().ok_or("Invalid scaling_mode string")?,
        *values.get(1).ok_or("Invalid scaling_mode string")?,
    ))
}

fn yuv_format_parser(s: &str) -> Result<PixelFormat, String> {
    match s {
        "420" => Ok(PixelFormat::Yuv420),
        "422" => Ok(PixelFormat::Yuv422),
        "444" => Ok(PixelFormat::Yuv444),
        "400" => Ok(PixelFormat::Yuv400),
        _ => Err(format!("Invalid yuv format: {s}")),
    }
}

fn codec_choice_parser(s: &str) -> Result<CodecChoice, String> {
    match s {
        "auto" | "default" => Ok(CodecChoice::default()),
        "aom" | "libaom" => Ok(CodecChoice::Aom),
        "mediacodec" => Ok(CodecChoice::MediaCodec),
        "dav1d" => Ok(CodecChoice::Dav1d),
        "gav1" | "libgav1" => Ok(CodecChoice::Libgav1),
        #[cfg(feature = "jpegxl")]
        "hxl" | "jxl" | "jpegxl" | "libjxl" => Ok(CodecChoice::Libjxl),
        #[cfg(feature = "native_av1")]
        "native" | "native_av1" => Ok(CodecChoice::Native),
        _ => Err(format!("Invalid codec choice: {s}")),
    }
}

fn header_format_parser(s: &str) -> Result<HeaderFormat, String> {
    match s {
        "meta" | "default" => Ok(HeaderFormat::Default),
        "mini" => Ok(HeaderFormat::Mini),
        _ => Err(format!("Invalid header format: {s}")),
    }
}

#[derive(Parser)]
struct CommandLineArgs {
    /// Prints the version of this crate and the version of each available codec
    #[arg(long, default_value = "false")]
    version: bool,

    /// Codec to use for encoding/decoding
    #[arg(short = 'c', long, value_parser = codec_choice_parser, default_value = "auto")]
    codec: CodecChoice,

    /// AVIF Decode only: Disable strict decoding, which disables strict validation checks and
    /// errors
    #[arg(long, default_value = "false")]
    no_strict: bool,

    /// AVIF Decode only: Decode all frames and display all image information instead of saving to
    /// disk
    #[arg(short = 'i', long, default_value = "false")]
    info: bool,

    /// Number of threads to use for AVIF encoding/decoding
    #[arg(long)]
    jobs: Option<u32>,

    /// AVIF Decode only:  When decoding an image sequence or progressive image, specify which
    /// frame index to decode (Default: 0)
    #[arg(long, short = 'I')]
    index: Option<u32>,

    /// Output depth, either 8 or 16. (AVIF/PNG only; For y4m/yuv, source depth is retained; JPEG
    /// is always 8bit)
    #[arg(long, short = 'd', value_parser = depth_parser)]
    depth: Option<u8>,

    /// Output quality in 0..100. (JPEG/AVIF only, default: 90).
    #[arg(long, short = 'q', value_parser = value_parser!(f32))]
    quality: Option<f32>,

    /// AVIF Encode only: Output alpha channel quality in 0..100. (default: quality).
    #[arg(long, value_parser = value_parser!(f32))]
    quality_alpha: Option<f32>,

    /// AVIF Encode only: Output gainmap quality in 0..100. (default: quality).
    #[arg(long, value_parser = value_parser!(f32))]
    quality_gainmap: Option<f32>,

    /// PNG output compression level in 0..9 (default: 5).
    #[arg(long, value_parser = value_parser!(i32).range(0..=9))]
    png_compress: Option<i32>,

    /// AVIF Encode only: Speed used for encoding.
    #[arg(long, short = 's', value_parser = value_parser!(u32).range(0..=10))]
    speed: Option<u32>,

    /// AVIF Encode only: MetaBox- or MinimizedImageBox-flavored HEIF file.
    #[arg(long, value_parser = header_format_parser, default_value = "meta")]
    header: HeaderFormat,

    /// AVIF Encode only: Write more metadata in the PixelInformationBox. Backward-compatible.
    #[arg(long, default_value = "false")]
    force_write_extended_pixi: bool,

    /// When decoding AVIF: Enable progressive AVIF processing. If a progressive image is
    /// encountered and --progressive is passed, --index will be used to choose which layer to
    /// decode (in progressive order).
    /// When encoding AVIF: Auto set parameters to encode a simple layered image supporting
    /// progressive rendering from a single input frame.
    #[arg(long, default_value = "false")]
    progressive: bool,

    /// Maximum image size (in total pixels) that should be tolerated (0 means unlimited)
    #[arg(long)]
    size_limit: Option<u32>,

    /// Maximum image dimension (width or height) that should be tolerated (0 means unlimited)
    #[arg(long)]
    dimension_limit: Option<u32>,

    /// If the input file contains embedded Exif metadata, ignore it (no-op if absent)
    #[arg(long, default_value = "false")]
    ignore_exif: bool,

    /// If the input file contains embedded XMP metadata, ignore it (no-op if absent)
    #[arg(long, default_value = "false")]
    ignore_xmp: bool,

    /// If the input file contains an embedded ICC color profile, ignore it (no-op if absent)
    #[arg(long, default_value = "false")]
    ignore_icc: bool,

    /// If the input file contains an alpha channel, ignore it (no-op if absent)
    #[arg(long, default_value = "false")]
    ignore_alpha: bool,

    /// AVIF Encode only: Add irot property (rotation), in 0..3. Makes (90 * ANGLE) degree rotation
    /// anti-clockwise
    #[arg(long = "irot", value_parser = value_parser!(u8).range(0..=3))]
    irot_angle: Option<u8>,

    /// AVIF Encode only: Add imir property (mirroring). 0=top-to-bottom, 1=left-to-right
    #[arg(long = "imir", value_parser = value_parser!(u8).range(0..=1))]
    imir_axis: Option<u8>,

    /// AVIF Encode only: Add clap property (clean aperture). Width, Height, HOffset, VOffset (in
    /// numerator/denominator pairs)
    #[arg(long, value_parser = clap_parser)]
    clap: Option<CleanAperture>,

    /// When encoding AVIF: Add clap property (clean aperture) calculated from a crop rectangle.
    /// When decoding AVIF: Only decode this region of the image (relative to its clean aperture if
    /// it has a valid one, before rotation and mirroring; with --extract-gainmap, relative to the
    /// gain map). X and Y are rounded down to even values along chroma-subsampled axes. X, Y,
    /// Width, Height
    #[arg(long, value_parser = crop_parser)]
    crop: Option<CropRect>,

    /// AVIF Decode only: Downscale the decoded image (or cropped region when --crop is also set,
    /// or clean aperture region if the image has a valid one; with --extract-gainmap, the gain
    /// map) to this size before rotation and mirroring. Width, Height
    #[arg(long, value_parser = target_size_parser)]
    target_size: Option<ImageSize>,

    /// AVIF Encode only: Add pasp property (aspect ratio). Horizontal spacing, Vertical spacing
    #[arg(long, value_parser = pasp_parser)]
    pasp: Option<PixelAspectRatio>,

    /// AVIF Encode only: Add clli property (content light level information). MaxCLL, MaxPALL
    #[arg(long, value_parser = clli_parser)]
    clli: Option<ContentLightLevelInformation>,

    /// AVIF Encode only: Set CICP values (nclx colr box) (P/T/M 3 raw numbers, use -r to set range
    /// flag)
    #[arg(long, value_parser = cicp_parser)]
    cicp: Option<Nclx>,

    /// AVIF Encode only: Provide an ICC profile payload to be associated with the primary item
    #[arg(long)]
    icc: Option<String>,

    /// AVIF Encode only: Provide an XMP metadata payload to be associated with the primary item
    #[arg(long)]
    xmp: Option<String>,

    /// AVIF Encode only: Provide an Exif metadata payload to be associated with the primary item
    #[arg(long)]
    exif: Option<String>,

    /// AVIF Encode only: Set frame scaling mode as given fraction
    #[arg(long, value_parser = scaling_mode_parser)]
    scaling_mode: Option<IFraction>,

    /// AVIF Encode only: log2 of number of tile rows
    #[arg(long, value_parser = value_parser!(i32).range(0..=6))]
    tilerowslog2: Option<i32>,

    /// AVIF Encode only: log2 of number of tile columns
    #[arg(long, value_parser = value_parser!(i32).range(0..=6))]
    tilecolslog2: Option<i32>,

    /// AVIF Encode only: Set tile rows and columns automatically. If specified, tilesrowslog2 and
    /// tilecolslog2 will be ignored
    #[arg(long, default_value = "false")]
    autotiling: bool,

    /// AVIF Encode only: Output format, one of 444, 422, 420 or 400. Ignored for y4m. For all
    /// other cases, auto defaults to 444.
    #[arg(long = "yuv", value_parser = yuv_format_parser)]
    yuv_format: Option<PixelFormat>,

    /// AVIF Encode only: Number of times an animated image sequence will be repeated, or
    /// 'infinite' for infinite repetitions. (Default: infinite)
    #[arg(long, default_value = "infinite", value_parser = repetition_count_parser)]
    repetition_count: RepetitionCount,

    /// AVIF Decode only: Allow sample transform. (Default: false)
    #[arg(long, default_value = "false")]
    allow_sample_transform: bool,

    /// AVIF Decode only: Decode and output only the gainmap.
    #[arg(long, default_value = "false")]
    extract_gainmap: bool,

    /// AVIF Decode only: Decode and output only the thumbnail.
    #[arg(long, default_value = "false")]
    extract_thumbnail: bool,

    /// AVIF Encode only: CLLI information of the alternate image. Only used when encoding an image
    /// with gainmap. Ignored otherwise.
    #[arg(long, value_parser = clli_parser)]
    alt_clli: Option<ContentLightLevelInformation>,

    /// AVIF Encode only: Encode a grid AVIF with M cols and N rows, expressed as MxN
    #[arg(short = 'g', long, value_parser = grid_parser)]
    grid: Option<GridSize>,

    /// Tone map an image with a gain map to --headroom
    #[arg(long, default_value = "false")]
    tonemap: bool,

    /// Target HDR headroom (log2(HDR/SDR), 0.0 means SDR) when using --tonemap
    #[arg(long, value_parser = value_parser!(f32), default_value = "0.0")]
    headroom: f32,

    /// Override output CICP values (P/T/M) when using --tonemap
    #[arg(long, value_parser = cicp_parser)]
    cicp_output: Option<Nclx>,

    /// Input AVIF file
    #[arg(allow_hyphen_values = false)]
    input_file: Option<String>,

    /// Output file
    #[arg(allow_hyphen_values = false)]
    output_file: Option<String>,
}

fn print_data_as_columns(rows: &[(usize, &str, String)]) {
    let rows: Vec<_> = rows
        .iter()
        .filter(|x| !x.1.is_empty())
        .map(|x| (format!("{} * {}", " ".repeat(x.0 * 4), x.1), x.2.as_str()))
        .collect();

    // Calculate the maximum width for the first column.
    let mut max_col1_width = 0;
    for (col1, _) in &rows {
        max_col1_width = max_col1_width.max(col1.len());
    }

    for (col1, col2) in &rows {
        println!("{col1:<max_col1_width$} : {col2}");
    }
}

fn print_vec(data: &[u8]) -> String {
    if data.is_empty() {
        "Absent".to_string()
    } else {
        format!("Present ({} bytes)", data.len())
    }
}

fn format_fraction_parts(num: i64, den: u32) -> String {
    let val = if den != 0 { num as f64 / f64::from(den) } else { 0.0 };
    format!("{val} (as fraction: {num}/{den})")
}

fn format_fraction_parts_triplet(fractions: [(i64, u32); 3]) -> String {
    let [r, g, b] = fractions.map(|(num, den)| format_fraction_parts(num, den));
    format!("R {r:<40} G {g:<40} B {b}")
}

fn format_fraction_triplet(fractions: [crabby_avif::utils::Fraction; 3]) -> String {
    format_fraction_parts_triplet(fractions.map(|f| (i64::from(f.0), f.1)))
}

fn gainmap_info_rows(
    gainmap: &crabby_avif::gainmap::GainMap,
    gainmap_decoded: bool,
) -> Vec<(usize, &'static str, String)> {
    let gainmap_image = &gainmap.image;
    let base_image = if gainmap.metadata.base_hdr_headroom.0 == 0 { "SDR" } else { "HDR" };
    vec![
        (
            0,
            "Gainmap",
            if gainmap_decoded {
                format!(
                    "{}x{} pixels, {} bit, {:#?}, {:#?} Range, Matrix Coeffs. {:#?}, Base Image is {}",
                    gainmap_image.width,
                    gainmap_image.height,
                    gainmap_image.depth,
                    gainmap_image.yuv_format,
                    gainmap_image.yuv_range,
                    gainmap_image.matrix_coefficients,
                    base_image,
                )
            } else {
                format!("Present (not decoded), Base Image is {base_image}")
            },
        ),
        (
            1,
            "Base headroom",
            format_fraction_parts(
                i64::from(gainmap.metadata.base_hdr_headroom.0),
                gainmap.metadata.base_hdr_headroom.1,
            ),
        ),
        (
            1,
            "Alternate headroom",
            format_fraction_parts(
                i64::from(gainmap.metadata.alternate_hdr_headroom.0),
                gainmap.metadata.alternate_hdr_headroom.1,
            ),
        ),
        (
            1,
            "Gain Map Min",
            format_fraction_triplet(gainmap.metadata.min),
        ),
        (
            1,
            "Gain Map Max",
            format_fraction_triplet(gainmap.metadata.max),
        ),
        (
            1,
            "Base Offset",
            format_fraction_triplet(gainmap.metadata.base_offset),
        ),
        (
            1,
            "Alternate Offset",
            format_fraction_triplet(gainmap.metadata.alternate_offset),
        ),
        (
            1,
            "Gain Map Gamma",
            format_fraction_parts_triplet(gainmap.metadata.gamma.map(|f| (i64::from(f.0), f.1))),
        ),
        (
            1,
            "Use Base Color Space",
            if gainmap.metadata.use_base_color_space { "True" } else { "False" }.to_string(),
        ),
        (0, "Alternate image", "".to_string()),
        (
            1,
            "Color Primaries",
            format!("{:#?}", gainmap.alt_color_primaries),
        ),
        (
            1,
            "Transfer Characteristics",
            format!("{:#?}", gainmap.alt_transfer_characteristics),
        ),
        (
            1,
            "Matrix Coefficients",
            format!("{:#?}", gainmap.alt_matrix_coefficients),
        ),
        (1, "ICC Profile", print_vec(&gainmap.alt_icc)),
        (1, "Bit Depth", format!("{}", gainmap.alt_plane_depth)),
        (1, "Planes", format!("{}", gainmap.alt_plane_count)),
        if gainmap.alt_clli.max_cll != 0 || gainmap.alt_clli.max_pall != 0 {
            (
                1,
                "CLLI",
                format!(
                    "{}, {}",
                    gainmap.alt_clli.max_cll, gainmap.alt_clli.max_pall
                ),
            )
        } else {
            (1, "", "".into())
        },
    ]
}

fn print_image_info(decoder: &Decoder) {
    let image = decoder.image().unwrap();
    let mut image_data = vec![
        (
            0,
            "File Format",
            format!("{:#?}", decoder.compression_format()),
        ),
        (0, "Resolution", format!("{}x{}", image.width, image.height)),
        (0, "Bit Depth", format!("{}", image.depth)),
        (0, "Format", format!("{:#?}", image.yuv_format)),
        if image.yuv_format == PixelFormat::Yuv420 {
            (
                0,
                "Chroma Sample Position",
                format!("{:#?}", image.chroma_sample_position),
            )
        } else {
            (0, "", "".into())
        },
        (
            0,
            "Alpha",
            match (image.alpha_present, image.alpha_premultiplied) {
                (true, true) => "Premultiplied".to_string(),
                (true, false) => "Not premultiplied".to_string(),
                (false, _) => "Absent".to_string(),
            },
        ),
        (0, "Range", format!("{:#?}", image.yuv_range)),
        (
            0,
            "Color Primaries",
            format!("{:#?}", image.color_primaries),
        ),
        (
            0,
            "Transfer Characteristics",
            format!("{:#?}", image.transfer_characteristics),
        ),
        (
            0,
            "Matrix Coefficients",
            format!("{:#?}", image.matrix_coefficients),
        ),
        (0, "ICC Profile", print_vec(&image.icc)),
        (0, "XMP Metadata", print_vec(&image.xmp)),
        (0, "Exif Metadata", print_vec(&image.exif)),
    ];
    if image.pasp.is_none()
        && image.clap.is_none()
        && image.irot_angle.is_none()
        && image.imir_axis.is_none()
    {
        image_data.push((0, "Transformations", "None".to_string()));
    } else {
        image_data.push((0, "Transformations", "".to_string()));
        if let Some(pasp) = image.pasp {
            image_data.push((
                1,
                "pasp (Aspect Ratio)",
                format!("{}/{}", pasp.h_spacing, pasp.v_spacing),
            ));
        }
        if let Some(clap) = image.clap {
            image_data.push((1, "clap (Clean Aperture)", "".to_string()));
            image_data.push((2, "W", format!("{}/{}", clap.width.0, clap.width.1)));
            image_data.push((2, "H", format!("{}/{}", clap.height.0, clap.height.1)));
            image_data.push((
                2,
                "hOff",
                format!("{}/{}", clap.horiz_off.0, clap.horiz_off.1),
            ));
            image_data.push((
                2,
                "vOff",
                format!("{}/{}", clap.vert_off.0, clap.vert_off.1),
            ));
            match CropRect::create_from(&clap, image.width, image.height, image.yuv_format) {
                Ok(rect) => image_data.extend_from_slice(&[
                    (2, "Valid, derived crop rect", "".to_string()),
                    (3, "X", format!("{}", rect.x)),
                    (3, "Y", format!("{}", rect.y)),
                    (3, "W", format!("{}", rect.width)),
                    (3, "H", format!("{}", rect.height)),
                ]),
                Err(_) => image_data.push((2, "Invalid", "".to_string())),
            }
        }
        if let Some(angle) = image.irot_angle {
            image_data.push((1, "irot (Rotation)", format!("{angle}")));
        }
        if let Some(axis) = image.imir_axis {
            image_data.push((1, "imir (Mirror)", format!("{axis}")));
        }
    }
    image_data.push((0, "Progressive", format!("{:#?}", image.progressive_state)));
    if let Some(clli) = image.clli {
        image_data.push((0, "CLLI", format!("{}, {}", clli.max_cll, clli.max_pall)));
    }
    if decoder.gainmap_present() {
        // The dimensions and format of the gain map image are only set if the gain map is decoded.
        let gainmap_decoded = matches!(
            decoder.settings.image_content_to_decode,
            ImageContentType::GainMap | ImageContentType::ColorAndGainMap | ImageContentType::All
        );
        image_data.extend(gainmap_info_rows(decoder.gainmap(), gainmap_decoded));
    } else {
        // TODO: b/394162563 - check if we need to report the present but ignored case.
        image_data.push((0, "Gainmap", "Absent".to_string()));
    }
    if image.image_sequence_track_present {
        image_data.push((
            0,
            "Repeat Count",
            match decoder.repetition_count() {
                RepetitionCount::Finite(x) => format!("{x}"),
                RepetitionCount::Infinite => "Infinite".to_string(),
                RepetitionCount::Unknown => "Unknown".to_string(),
            },
        ));
    }
    print_data_as_columns(&image_data);
}

fn max_threads(jobs: &Option<u32>) -> u32 {
    match jobs {
        Some(x) => {
            if *x == 0 {
                match std::thread::available_parallelism() {
                    Ok(value) => value.get() as u32,
                    Err(_) => 1,
                }
            } else {
                *x
            }
        }
        None => 1,
    }
}

fn create_decoder_and_parse(
    args: &CommandLineArgs,
    input_file: &String,
    allow_progressive: bool,
) -> AvifResult<Decoder> {
    let mut settings = decoder::Settings {
        source: if args.extract_thumbnail {
            decoder::Source::Thumbnail
        } else {
            decoder::Source::Auto
        },
        strictness: if args.no_strict { Strictness::None } else { Strictness::All },
        image_content_to_decode: if args.extract_gainmap {
            ImageContentType::GainMap
        } else if args.crop.is_some() || args.target_size.is_some() {
            // Region and scaled decodes cannot decode both the color image and the gain map. The
            // gain map is only written with --extract-gainmap anyway.
            if args.ignore_alpha {
                ImageContentType::Color
            } else {
                ImageContentType::ColorAndAlpha
            }
        } else if args.ignore_alpha {
            ImageContentType::ColorAndGainMap
        } else {
            ImageContentType::All
        },
        codec_choice: args.codec,
        max_threads: max_threads(&args.jobs),
        allow_progressive,
        ignore_exif: args.ignore_exif,
        ignore_xmp: args.ignore_xmp,
        allow_sample_transform: args.allow_sample_transform,
        ..Default::default()
    };
    // These values cannot be initialized in the list above since we need the default values to be
    // retain unless they are explicitly specified.
    if let Some(size_limit) = args.size_limit {
        settings.image_size_limit = NonZero::new(size_limit);
    }
    if let Some(dimension_limit) = args.dimension_limit {
        settings.image_dimension_limit = NonZero::new(dimension_limit);
    }
    let mut decoder = Decoder::default();
    decoder.settings = settings;
    decoder
        .set_io_file(input_file)
        .or(Err(AvifError::UnknownError(
            "Cannot open input file".into(),
        )))?;
    decoder.parse()?;
    Ok(decoder)
}

fn info(args: &CommandLineArgs, input_file: &String) -> AvifResult<()> {
    let mut decoder = create_decoder_and_parse(args, input_file, args.progressive)?;
    println!("Image decoded: {}", input_file);
    print_image_info(&decoder);
    println!(
        " * {} timescales per second, {} seconds ({} timescales), {} frame{}",
        decoder.timescale(),
        decoder.duration(),
        decoder.duration_in_timescales(),
        decoder.image_count(),
        if decoder.image_count() == 1 { "" } else { "s" },
    );
    if decoder.image_count() > 1 {
        let image = decoder.image().unwrap();
        println!(
            " * {} Frames: ({} expected frames)",
            if image.image_sequence_track_present {
                "Image Sequence"
            } else {
                "Progressive Image"
            },
            decoder.image_count()
        );
    } else {
        println!(" * Frame:");
    }

    let mut index = 0;
    loop {
        match decoder.next_image() {
            Ok(_) => {
                println!("     * Decoded frame [{}] [pts {} ({} timescales)] [duration {} ({} timescales)] [{}x{}]",
                    index,
                    decoder.image_timing().pts,
                    decoder.image_timing().pts_in_timescales,
                    decoder.image_timing().duration,
                    decoder.image_timing().duration_in_timescales,
                    decoder.image().unwrap().width,
                    decoder.image().unwrap().height);
                index += 1;
            }
            Err(AvifError::NoImagesRemaining) => {
                return Ok(());
            }
            Err(err) => {
                return Err(err);
            }
        }
    }
}

fn get_extension(filename: &str) -> String {
    std::path::Path::new(filename)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
}

fn decode(args: &CommandLineArgs, input_file: &String) -> AvifResult<()> {
    let max_threads = max_threads(&args.jobs);
    println!(
        "Decoding with {max_threads} worker thread{}, please wait...",
        if max_threads == 1 { "" } else { "s" }
    );
    let mut decoder = create_decoder_and_parse(args, input_file, args.progressive)?;
    if args.extract_gainmap && !decoder.gainmap_present() {
        return Err(AvifError::UnknownError(
            "Input image does not contain a gain map".into(),
        ));
    }
    let options = DecodeOptions {
        crop_rect: args.crop,
        scaled_size: args.target_size,
    };
    match (
        decoder.nth_image_with_options(args.index.unwrap_or(0), &options),
        args.crop.is_some(),
        args.target_size.is_some(),
    ) {
        (Err(AvifError::InvalidArgument), true, false) => {
            return Err(AvifError::UnknownError(
                "Invalid crop rectangle: it must fit in the image (or in its clean aperture)"
                    .into(),
            ));
        }
        (Err(AvifError::InvalidArgument), false, true) => {
            return Err(AvifError::UnknownError(
                "Invalid target size: width and height must be non-zero and not exceed the image (or clean aperture) dimensions"
                    .into(),
            ));
        }
        (Err(AvifError::InvalidArgument), true, true) => {
            return Err(AvifError::UnknownError(
                "Invalid crop rectangle or target size: crop must fit in the image (or clean aperture) and target size must be non-zero and not exceed the crop dimensions"
                    .into(),
            ));
        }
        (result, _, _) => result?,
    }
    println!("Image Decoded: {}", input_file);
    println!("Image details:");
    print_image_info(&decoder);

    let output_filename = &args.output_file.as_ref().unwrap().as_str();
    let image = if args.extract_gainmap {
        &decoder.gainmap().image
    } else {
        decoder.image().unwrap()
    };
    let mut owned_image = if image.clap.is_some() {
        match image.cropped_image() {
            Ok(cropped_image) => Some(cropped_image),
            Err(_) => {
                println!("Warning: clap was invalid. So writing whole image.");
                None
            }
        }
    } else {
        None
    };
    let strip_icc = args.ignore_icc && !image.icc.is_empty();
    let strip_alpha = args.ignore_alpha && (image.alpha_present || image.has_alpha());
    if strip_icc || strip_alpha {
        let target = match &mut owned_image {
            Some(img) => img,
            None => owned_image.insert(image.try_deep_clone()?),
        };
        if strip_icc {
            target.icc.clear();
        }
        if strip_alpha {
            clear_alpha(target);
        }
    }
    let image_to_write = owned_image.as_ref().unwrap_or(image);
    let extension = get_extension(output_filename);
    if can_encode(output_filename) {
        return write_decoded_avif(args, image, owned_image, output_filename);
    }
    write_non_avif_image(args, image_to_write, output_filename, &extension)?;
    println!(
        "Wrote image at index {} to output {}",
        args.index.unwrap_or(0),
        output_filename,
    );
    Ok(())
}

fn write_non_avif_image(
    #[cfg_attr(not(any(feature = "png", feature = "jpeg")), allow(unused_variables))]
    args: &CommandLineArgs,
    image: &image::Image,
    output_filename: &str,
    extension: &str,
) -> AvifResult<()> {
    let mut writer: Box<dyn Writer> = match extension {
        "y4m" | "yuv" => {
            if !image.icc.is_empty() || !image.exif.is_empty() || !image.xmp.is_empty() {
                println!("Warning: metadata dropped when saving to {extension}");
            }
            Box::new(Y4MWriter::create(extension == "yuv"))
        }
        #[cfg(feature = "png")]
        "png" => Box::new(PngWriter {
            depth: args.depth,
            compression_level: args.png_compress,
        }),
        #[cfg(feature = "jpeg")]
        "jpg" | "jpeg" => Box::new(JpegWriter {
            quality: args.quality.map(|quality| quality as u8),
        }),
        _ => {
            return Err(AvifError::UnknownError(format!(
                "Unknown output file extension ({extension})"
            )));
        }
    };
    let mut output_file = File::create(output_filename).or(Err(AvifError::UnknownError(
        "Could not open output file".into(),
    )))?;
    writer.write_frame(&mut output_file, image)
}

fn read_image_with_optional_gainmap(
    args: &CommandLineArgs,
    input_file: &str,
    cicp_override: Option<&Nclx>,
) -> AvifResult<(image::Image, Option<crabby_avif::gainmap::GainMap>)> {
    let (mut image, mut gainmap) = if can_decode(input_file) {
        let input_string = input_file.to_string();
        let mut decoder = create_decoder_and_parse(args, &input_string, false)?;
        decoder.nth_image(args.index.unwrap_or(0))?;
        let image = decoder
            .image()
            .ok_or(AvifError::NoContent)?
            .try_deep_clone()?;
        let gainmap = if decoder.gainmap_present() {
            Some(decoder.gainmap().try_deep_clone()?)
        } else {
            None
        };
        (image, gainmap)
    } else {
        #[cfg(feature = "encoder")]
        {
            let mut reader = <dyn Reader>::create(input_file)?;
            let reader_config = Config {
                // Read at full native precision (4:4:4 for JPEG) instead of output --yuv/--depth.
                yuv_format: match get_extension(input_file).as_str() {
                    "jpg" | "jpeg" => Some(PixelFormat::Yuv444),
                    _ => None,
                },
                depth: None,
                matrix_coefficients: cicp_override.map(|n| n.matrix_coefficients),
                ignore_icc: args.ignore_icc,
                ignore_exif: args.ignore_exif,
                ignore_xmp: args.ignore_xmp,
                allow_sample_transform: args.allow_sample_transform,
                ..Default::default()
            };
            let (image, _, gainmap) = reader.read_frame(&reader_config)?;
            (image, gainmap)
        }
        #[cfg(not(feature = "encoder"))]
        {
            return Err(AvifError::NotImplemented);
        }
    };
    apply_read_ignore_flags(args, &mut image, gainmap.as_mut());
    if let Some(nclx) = cicp_override {
        image.color_primaries = nclx.color_primaries;
        image.transfer_characteristics = nclx.transfer_characteristics;
        image.matrix_coefficients = nclx.matrix_coefficients;
    } else if image.icc.is_empty()
        && image.color_primaries == ColorPrimaries::Unspecified
        && image.transfer_characteristics == TransferCharacteristics::Unspecified
    {
        image.color_primaries = ColorPrimaries::Srgb;
        image.transfer_characteristics = TransferCharacteristics::Srgb;
    }
    Ok((image, gainmap))
}

fn tonemap(args: &CommandLineArgs, input_file: &str, output_file: &str) -> AvifResult<()> {
    let (image, gainmap) = read_image_with_optional_gainmap(args, input_file, args.cicp.as_ref())?;
    let gainmap = gainmap
        .filter(|gm| gm.image.width > 0 && gm.image.height > 0)
        .ok_or(AvifError::InvalidArgument)?;
    if !image.icc.is_empty() || !gainmap.alt_icc.is_empty() {
        println!("Tone mapping for images with ICC profiles is not supported; pass --ignore-icc");
        return Err(AvifError::NotImplemented);
    }
    gainmap::validate_gainmap_metadata(&gainmap.metadata)?;

    let headroom = args.headroom;
    let tone_mapping_to_hdr = headroom > 0.0;
    let base_headroom =
        gainmap.metadata.base_hdr_headroom.0 as f32 / gainmap.metadata.base_hdr_headroom.1 as f32;
    let alt_headroom = gainmap.metadata.alternate_hdr_headroom.0 as f32
        / gainmap.metadata.alternate_hdr_headroom.1 as f32;
    let tone_mapping_to_base = (headroom <= base_headroom && base_headroom <= alt_headroom)
        || (headroom >= base_headroom && base_headroom >= alt_headroom);
    let tone_mapping_to_alt = (headroom <= alt_headroom && alt_headroom <= base_headroom)
        || (headroom >= alt_headroom && alt_headroom >= base_headroom);
    let base_is_hdr = base_headroom != 0.0;

    let (mut out_cp, mut out_tc, out_mc) = if let Some(nclx) = &args.cicp_output {
        (
            nclx.color_primaries,
            nclx.transfer_characteristics,
            nclx.matrix_coefficients,
        )
    } else if tone_mapping_to_base || (tone_mapping_to_hdr && base_is_hdr) {
        (
            image.color_primaries,
            image.transfer_characteristics,
            image.matrix_coefficients,
        )
    } else {
        (
            gainmap.alt_color_primaries,
            gainmap.alt_transfer_characteristics,
            gainmap.alt_matrix_coefficients,
        )
    };
    if out_cp == ColorPrimaries::Unspecified {
        out_cp = if gainmap.metadata.use_base_color_space
            || gainmap.alt_color_primaries == ColorPrimaries::Unspecified
        {
            image.color_primaries
        } else {
            gainmap.alt_color_primaries
        };
    }
    if out_tc == TransferCharacteristics::Unspecified {
        out_tc = if tone_mapping_to_hdr {
            TransferCharacteristics::Pq
        } else {
            TransferCharacteristics::Srgb
        };
    }

    if !matches!(image.depth, 8 | 10 | 12 | 16) || !matches!(gainmap.image.depth, 8 | 10 | 12 | 16)
    {
        return Err(AvifError::NotImplemented);
    }
    let alt_depth = if matches!(gainmap.alt_plane_depth, 8 | 10 | 12 | 16) {
        gainmap.alt_plane_depth
    } else {
        0
    };
    let requested_depth = args.depth.unwrap_or_else(|| {
        let d = if tone_mapping_to_base {
            image.depth
        } else if tone_mapping_to_alt {
            alt_depth
        } else {
            0
        };
        if d == 0 {
            image.depth.max(gainmap.image.depth).max(alt_depth)
        } else {
            d
        }
    });
    let out_depth = if can_encode(output_file) { requested_depth.min(12) } else { requested_depth };

    let less_subsampled = if matches!(
        (image.yuv_format, gainmap.image.yuv_format),
        (PixelFormat::Yuv444, _) | (_, PixelFormat::Yuv444)
    ) {
        PixelFormat::Yuv444
    } else if matches!(
        (image.yuv_format, gainmap.image.yuv_format),
        (PixelFormat::Yuv422, _) | (_, PixelFormat::Yuv422)
    ) {
        PixelFormat::Yuv422
    } else if matches!(
        (image.yuv_format, gainmap.image.yuv_format),
        (PixelFormat::Yuv420, _) | (_, PixelFormat::Yuv420)
    ) {
        PixelFormat::Yuv420
    } else {
        PixelFormat::Yuv400
    };
    let yuv_format = args.yuv_format.unwrap_or(if tone_mapping_to_base {
        image.yuv_format
    } else if tone_mapping_to_alt && gainmap.alt_plane_count == 1 {
        PixelFormat::Yuv400
    } else {
        less_subsampled
    });
    let mut out_mc = if out_mc == MatrixCoefficients::Unspecified {
        MatrixCoefficients::Bt601
    } else {
        out_mc
    };
    let unsupported_mc = matches!(
        out_mc,
        MatrixCoefficients::Reserved
            | MatrixCoefficients::Bt2020Cl
            | MatrixCoefficients::Smpte2085
            | MatrixCoefficients::ChromaDerivedCl
            | MatrixCoefficients::Ictcp
            | MatrixCoefficients::YcgcoRe
            | MatrixCoefficients::YcgcoRo
    ) || (out_mc == MatrixCoefficients::Ycgco
        && image.yuv_range == image::YuvRange::Limited)
        || (out_mc == MatrixCoefficients::Identity && yuv_format != PixelFormat::Yuv444);
    if unsupported_mc {
        if args.cicp_output.is_some() {
            return Err(AvifError::NotImplemented);
        }
        out_mc = MatrixCoefficients::Bt601;
    }

    let (tm_rgb, computed_clli) =
        gainmap::apply_gainmap(&image, &gainmap, headroom, out_cp, out_tc, out_depth)?;

    let clli = if args.clli.is_some() {
        args.clli
    } else if tone_mapping_to_base
        && image
            .clli
            .is_some_and(|c| c.max_cll != 0 || c.max_pall != 0)
    {
        image.clli
    } else if tone_mapping_to_alt
        && (gainmap.alt_clli.max_cll != 0 || gainmap.alt_clli.max_pall != 0)
    {
        Some(gainmap.alt_clli)
    } else if tone_mapping_to_hdr && (computed_clli.max_cll != 0 || computed_clli.max_pall != 0) {
        Some(computed_clli)
    } else {
        None
    };

    let has_alpha = image.has_alpha();
    let mut tone_mapped = image::Image {
        width: image.width,
        height: image.height,
        depth: out_depth,
        yuv_format,
        yuv_range: image.yuv_range,
        chroma_sample_position: ChromaSamplePosition::Unknown,
        alpha_present: has_alpha,
        alpha_premultiplied: has_alpha && image.alpha_premultiplied,
        color_primaries: out_cp,
        transfer_characteristics: out_tc,
        matrix_coefficients: out_mc,
        clli,
        pasp: image.pasp,
        clap: image.clap,
        irot_angle: image.irot_angle,
        imir_axis: image.imir_axis,
        exif: try_clone_bytes(&image.exif)?,
        icc: Vec::new(),
        xmp: try_clone_bytes(&image.xmp)?,
        ..Default::default()
    };
    tm_rgb.convert_to_yuv(&mut tone_mapped)?;

    let extension = get_extension(output_file);
    if can_encode(output_file) {
        encode_avif_from_args_codec(args, &tone_mapped, None, output_file)
    } else {
        write_non_avif_image(args, &tone_mapped, output_file, &extension)?;
        println!("Wrote tone mapped image to output {output_file}");
        Ok(())
    }
}

#[cfg(feature = "encoder")]
fn write_decoded_avif(
    args: &CommandLineArgs,
    image: &image::Image,
    mut owned_image: Option<image::Image>,
    output_filename: &str,
) -> AvifResult<()> {
    let src = owned_image.as_ref().unwrap_or(image);
    let converted = convert_image_format_and_depth(
        src,
        args.yuv_format,
        args.depth,
        args.cicp.as_ref(),
        false,
        false,
        args.cicp.is_none(),
    )?;
    let needs_metadata_fixup = src.clap.is_some()
        || (args.cicp.is_none()
            && src.icc.is_empty()
            && src.color_primaries == ColorPrimaries::Unspecified
            && src.transfer_characteristics == TransferCharacteristics::Unspecified);
    let mut target = converted.or_else(|| owned_image.take());
    let image_for_avif = if target.is_some() || needs_metadata_fixup {
        let img = match &mut target {
            Some(img) => img,
            None => target.insert(image.try_deep_clone()?),
        };
        // decode() already applied or ignored clap, so do not re-emit stale clap on the encoded AVIF.
        img.clap = None;
        if args.cicp.is_none()
            && img.icc.is_empty()
            && img.color_primaries == ColorPrimaries::Unspecified
            && img.transfer_characteristics == TransferCharacteristics::Unspecified
        {
            img.color_primaries = ColorPrimaries::Srgb;
            img.transfer_characteristics = TransferCharacteristics::Srgb;
        }
        &*img
    } else {
        image
    };
    // decode() uses args.codec to choose the decoder; pick the encoder codec from output_filename.
    let codec_choice = encoder_codec_choice_for_output(CodecChoice::Auto, output_filename);
    encode_single_image_to_avif(args, codec_choice, image_for_avif, None, output_filename)?;
    println!(
        "Wrote image at index {} to output {}",
        args.index.unwrap_or(0),
        output_filename,
    );
    Ok(())
}

#[cfg(not(feature = "encoder"))]
fn write_decoded_avif(
    _args: &CommandLineArgs,
    _image: &image::Image,
    _owned_image: Option<image::Image>,
    _output_filename: &str,
) -> AvifResult<()> {
    Err(AvifError::UnknownError(
        "AVIF output requires building with the encoder feature".into(),
    ))
}

fn clear_alpha(image: &mut image::Image) {
    // Match libavif behavior: drop the alpha plane and flags, leaving color planes as-is.
    image.planes[image::Plane::A as usize] = None;
    image.row_bytes[image::Plane::A as usize] = 0;
    image.alpha_present = false;
    image.alpha_premultiplied = false;
}

#[cfg(feature = "encoder")]
fn read_file(filepath: &String) -> io::Result<Vec<u8>> {
    let mut file = File::open(filepath)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn apply_read_ignore_flags(
    args: &CommandLineArgs,
    image: &mut image::Image,
    gainmap: Option<&mut crabby_avif::gainmap::GainMap>,
) {
    if args.ignore_icc {
        image.icc.clear();
        if let Some(gainmap) = gainmap {
            gainmap.alt_icc.clear();
        }
    }
    if args.ignore_alpha {
        clear_alpha(image);
    }
}

#[cfg(feature = "encoder")]
fn best_cell_size(num_pixels: u32, num_cells: u32, is_subsampled: bool) -> AvifResult<u32> {
    if num_cells == 0 {
        return Err(AvifError::InvalidImageGrid(
            "grid dimension must be positive".into(),
        ));
    }
    let mut cell_size = num_pixels.div_ceil(num_cells);
    if cell_size < 64 {
        cell_size = 64;
    }
    if cell_size > 65536 {
        return Err(AvifError::InvalidImageGrid(format!(
            "cell size {cell_size} exceeds maximum frame size 65536"
        )));
    }
    if is_subsampled && !cell_size.is_multiple_of(2) {
        cell_size += 1;
    }
    if u64::from(num_cells - 1) * u64::from(cell_size) >= u64::from(num_pixels) {
        return Err(AvifError::InvalidImageGrid(format!(
            "cannot split {num_pixels} pixels into {num_cells} cells (cell size {cell_size})"
        )));
    }
    Ok(cell_size)
}

#[cfg(feature = "encoder")]
fn add_image_single_or_grid(
    encoder: &mut Encoder,
    image: &image::Image,
    gainmap: Option<&crabby_avif::gainmap::GainMap>,
    grid: Option<GridSize>,
) -> AvifResult<()> {
    if let Some(GridSize { cols, rows }) = grid {
        if cols > 1 || rows > 1 {
            let cell_width =
                best_cell_size(image.width, cols, image.yuv_format.chroma_shift_x().0 != 0)?;
            let cell_height =
                best_cell_size(image.height, rows, image.yuv_format.chroma_shift_y() != 0)?;
            return match gainmap {
                Some(gainmap) => {
                    encoder.add_image_gainmap_split_grid(cell_width, cell_height, image, gainmap)
                }
                None => encoder.add_image_split_grid(cell_width, cell_height, image),
            };
        }
    }
    match gainmap {
        Some(gainmap) => encoder.add_image_gainmap(image, gainmap),
        None => encoder.add_image(image),
    }
}

#[cfg(feature = "encoder")]
const DEFAULT_ENCODE_QUALITY: f32 = 90.0;

#[cfg(feature = "encoder")]
fn apply_encode_overrides(
    args: &CommandLineArgs,
    image: &mut image::Image,
    gainmap: Option<&mut crabby_avif::gainmap::GainMap>,
) -> AvifResult<()> {
    if args.irot_angle.is_some() {
        image.irot_angle = args.irot_angle;
    }
    if args.imir_axis.is_some() {
        image.imir_axis = args.imir_axis;
    }
    if let Some(clap) = args.clap {
        image.clap = Some(clap);
    }
    if let Some(crop) = args.crop {
        image.clap = Some(CleanAperture::create_from(
            &crop,
            image.width,
            image.height,
            image.yuv_format,
        )?);
    }
    if args.pasp.is_some() {
        image.pasp = args.pasp;
    }
    if args.clli.is_some() {
        image.clli = args.clli;
    }
    if let Some(nclx) = &args.cicp {
        image.color_primaries = nclx.color_primaries;
        image.transfer_characteristics = nclx.transfer_characteristics;
        image.matrix_coefficients = nclx.matrix_coefficients;
    }
    if let Some(icc) = &args.icc {
        image.icc = read_file(icc).expect("failed to read icc file");
    }
    if let Some(exif) = &args.exif {
        image.exif = read_file(exif).expect("failed to read exif file");
    }
    if let Some(xmp) = &args.xmp {
        image.xmp = read_file(xmp).expect("failed to read xmp file");
    }
    if image.icc.is_empty()
        && args.cicp.is_none()
        && image.color_primaries == ColorPrimaries::Unspecified
        && image.transfer_characteristics == TransferCharacteristics::Unspecified
    {
        // The final image has no ICC profile, the user didn't specify any CICP, and the source
        // image didn't provide any CICP. Explicitly signal SRGB CP/TC here, as 2/2/x will be
        // interpreted as SRGB anyway.
        image.color_primaries = ColorPrimaries::Srgb;
        image.transfer_characteristics = TransferCharacteristics::Srgb;
    }
    if let Some(gainmap) = gainmap {
        if gainmap.alt_icc.is_empty() && gainmap.alt_color_primaries == ColorPrimaries::Unspecified
        {
            gainmap.alt_color_primaries = image.color_primaries;
        }
        if let Some(alt_clli) = args.alt_clli {
            gainmap.alt_clli = alt_clli;
        }
    }
    Ok(())
}

#[cfg(feature = "encoder")]
fn encoder_codec_choice_for_output(args_codec: CodecChoice, output_file: &str) -> CodecChoice {
    match args_codec {
        CodecChoice::Auto => match get_extension(output_file).as_str() {
            "avif" => CodecChoice::Auto, // CodecChoice::Auto maps to AV1 only. Keep it.
            #[cfg(feature = "jpegxl")]
            "hxl" => CodecChoice::Libjxl,
            _ => unreachable!(), // See can_encode().
        },
        explicit_codec_choice => explicit_codec_choice,
    }
}

#[cfg(feature = "encoder")]
fn create_encoder_settings(args: &CommandLineArgs, codec_choice: CodecChoice) -> encoder::Settings {
    let quality = args.quality.unwrap_or(DEFAULT_ENCODE_QUALITY);
    let mut settings = encoder::Settings {
        extra_layer_count: if args.progressive { 1 } else { 0 },
        codec_choice,
        speed: args.speed,
        header_format: args.header,
        force_write_extended_pixi: args.force_write_extended_pixi,
        timescale: 1000, // ms.
        repetition_count: args.repetition_count,
        mutable: MutableSettings {
            quality,
            quality_alpha: args.quality_alpha.unwrap_or(quality),
            quality_gainmap: args.quality_gainmap.unwrap_or(quality),
            tiling_mode: if args.autotiling {
                TilingMode::Auto
            } else {
                TilingMode::Manual(
                    args.tilerowslog2.unwrap_or(0),
                    args.tilecolslog2.unwrap_or(0),
                )
            },
            ..Default::default()
        },
        ..Default::default()
    };
    if let Some(scaling_mode) = args.scaling_mode {
        settings.mutable.scaling_mode = ScalingMode {
            horizontal: scaling_mode,
            vertical: scaling_mode,
        };
    }
    settings
}

#[cfg(feature = "encoder")]
fn write_encoded_avif(mut encoder: Encoder, output_file: &str) -> AvifResult<()> {
    let encoded_data = encoder.finish()?;
    let mut file = File::create(output_file).expect("file creation failed");
    file.write_all(&encoded_data).expect("file writing failed");
    println!("Write output AVIF: {output_file}");
    Ok(())
}

#[cfg(feature = "encoder")]
fn encode_single_image_to_avif(
    args: &CommandLineArgs,
    codec_choice: CodecChoice,
    image: &image::Image,
    gainmap: Option<&crabby_avif::gainmap::GainMap>,
    output_file: &str,
) -> AvifResult<()> {
    let mut settings = create_encoder_settings(args, codec_choice);
    settings.extra_layer_count = 0;
    let mut encoder = Encoder::create_with_settings(&settings)?;
    add_image_single_or_grid(&mut encoder, image, gainmap, args.grid)?;
    write_encoded_avif(encoder, output_file)
}

#[cfg(feature = "encoder")]
fn encode_avif_from_args_codec(
    args: &CommandLineArgs,
    image: &image::Image,
    gainmap: Option<&crabby_avif::gainmap::GainMap>,
    output_file: &str,
) -> AvifResult<()> {
    let encoder_arg_codec = match args.codec {
        CodecChoice::Aom => CodecChoice::Aom,
        #[cfg(feature = "avm")]
        CodecChoice::Avm => CodecChoice::Avm,
        #[cfg(feature = "jpegxl")]
        CodecChoice::Libjxl => CodecChoice::Libjxl,
        _ => CodecChoice::Auto,
    };
    let codec_choice = encoder_codec_choice_for_output(encoder_arg_codec, output_file);
    encode_single_image_to_avif(args, codec_choice, image, gainmap, output_file)
}

#[cfg(not(feature = "encoder"))]
fn encode_avif_from_args_codec(
    _args: &CommandLineArgs,
    _image: &image::Image,
    _gainmap: Option<&crabby_avif::gainmap::GainMap>,
    _output_file: &str,
) -> AvifResult<()> {
    Err(AvifError::UnknownError(
        "AVIF output requires building with the encoder feature".into(),
    ))
}

fn try_clone_bytes(src: &[u8]) -> AvifResult<Vec<u8>> {
    let mut dst = Vec::new();
    dst.try_reserve_exact(src.len())
        .map_err(|_| AvifError::OutOfMemory)?;
    dst.extend_from_slice(src);
    Ok(dst)
}

#[cfg(feature = "encoder")]
fn convert_image_format_and_depth(
    src: &image::Image,
    yuv_format: Option<PixelFormat>,
    depth: Option<u8>,
    target_cicp: Option<&Nclx>,
    ignore_icc: bool,
    ignore_alpha: bool,
    allow_mc_fallback: bool,
) -> AvifResult<Option<image::Image>> {
    if !matches!(src.depth, 8 | 10 | 12 | 16) {
        return Err(AvifError::NotImplemented);
    }
    let target_format = yuv_format.unwrap_or(src.yuv_format);
    let target_depth = depth.unwrap_or_else(|| src.depth.min(12));
    let target_cp = target_cicp.map_or(src.color_primaries, |n| n.color_primaries);
    let target_tc =
        target_cicp.map_or(src.transfer_characteristics, |n| n.transfer_characteristics);
    let mut target_mc = target_cicp.map_or(src.matrix_coefficients, |n| n.matrix_coefficients);
    if !ignore_icc
        && !src.icc.is_empty()
        && ((src.yuv_format == PixelFormat::Yuv400) != (target_format == PixelFormat::Yuv400))
    {
        return Err(AvifError::UnknownError(
            "Image contains an ICC profile that is incompatible with the requested output. Pass --ignore-icc to discard the ICC profile"
                .into(),
        ));
    }
    if target_mc == MatrixCoefficients::Identity && target_format != PixelFormat::Yuv444 {
        if allow_mc_fallback {
            target_mc = MatrixCoefficients::Bt601;
        } else {
            return Err(AvifError::UnknownError(
                "Identity matrix requires YUV 4:4:4; pass --cicp to convert matrix coefficients"
                    .into(),
            ));
        }
    }
    if target_mc == MatrixCoefficients::YcgcoRe
        && ((src.matrix_coefficients == MatrixCoefficients::YcgcoRe && target_depth != src.depth)
            || target_format == PixelFormat::Yuv400)
    {
        if allow_mc_fallback {
            target_mc = MatrixCoefficients::Bt601;
        } else {
            return Err(AvifError::UnknownError(
                "YCgCo-Re requires matching depth and color YUV format; pass --cicp to convert matrix coefficients"
                    .into(),
            ));
        }
    }
    if target_format == src.yuv_format
        && target_depth == src.depth
        && target_mc == src.matrix_coefficients
        && (target_mc != MatrixCoefficients::ChromaDerivedNcl || target_cp == src.color_primaries)
    {
        if target_cp == src.color_primaries
            && target_tc == src.transfer_characteristics
            && (!ignore_icc || src.icc.is_empty())
            && (!ignore_alpha || (!src.alpha_present && !src.has_alpha()))
        {
            return Ok(None);
        }
        let mut dst = src.try_deep_clone()?;
        dst.color_primaries = target_cp;
        dst.transfer_characteristics = target_tc;
        if ignore_icc {
            dst.icc.clear();
        }
        if ignore_alpha {
            clear_alpha(&mut dst);
        }
        return Ok(Some(dst));
    }
    let rgb_depth = match src.matrix_coefficients {
        MatrixCoefficients::YcgcoRe => {
            if !matches!(src.depth, 10 | 12) {
                return Err(AvifError::NotImplemented);
            }
            src.depth - 2
        }
        MatrixCoefficients::YcgcoRo => return Err(AvifError::NotImplemented),
        _ => src.depth,
    };
    if matches!(
        target_mc,
        MatrixCoefficients::Reserved
            | MatrixCoefficients::Bt2020Cl
            | MatrixCoefficients::Smpte2085
            | MatrixCoefficients::ChromaDerivedCl
            | MatrixCoefficients::Ictcp
    ) {
        return Err(AvifError::NotImplemented);
    }
    if target_mc == MatrixCoefficients::Ycgco && src.yuv_range == image::YuvRange::Limited {
        return Err(AvifError::NotImplemented);
    }
    if matches!(
        target_mc,
        MatrixCoefficients::YcgcoRe | MatrixCoefficients::YcgcoRo
    ) {
        let bit_offset = if target_mc == MatrixCoefficients::YcgcoRe { 2 } else { 1 };
        if src.yuv_range == image::YuvRange::Limited
            || target_depth.checked_sub(bit_offset) != Some(rgb_depth)
        {
            return Err(AvifError::NotImplemented);
        }
    }
    let src_without_alpha;
    let src = if ignore_alpha && src.has_alpha() {
        let mut cloned = src.try_deep_clone()?;
        clear_alpha(&mut cloned);
        src_without_alpha = cloned;
        &src_without_alpha
    } else {
        src
    };
    let has_alpha = src.has_alpha();
    let mut rgb = reformat::rgb::Image::create_from_yuv(src);
    rgb.depth = rgb_depth;
    rgb.premultiply_alpha = has_alpha && src.alpha_premultiplied;
    rgb.format = if has_alpha { reformat::rgb::Format::Rgba } else { reformat::rgb::Format::Rgb };
    rgb.allocate()?;
    rgb.convert_from_yuv(src)?;
    let mut dst = image::Image {
        width: src.width,
        height: src.height,
        depth: target_depth,
        yuv_format: target_format,
        yuv_range: src.yuv_range,
        alpha_present: has_alpha,
        alpha_premultiplied: has_alpha && src.alpha_premultiplied,
        color_primaries: target_cp,
        transfer_characteristics: target_tc,
        matrix_coefficients: target_mc,
        clli: src.clli,
        pasp: src.pasp,
        clap: src.clap,
        irot_angle: src.irot_angle,
        imir_axis: src.imir_axis,
        exif: try_clone_bytes(&src.exif)?,
        icc: if ignore_icc { Vec::new() } else { try_clone_bytes(&src.icc)? },
        xmp: try_clone_bytes(&src.xmp)?,
        ..Default::default()
    };
    rgb.convert_to_yuv(&mut dst)?;
    Ok(Some(dst))
}

#[cfg(feature = "encoder")]
fn reencode(args: &CommandLineArgs, input_file: &String, output_file: &str) -> AvifResult<()> {
    let mut decoder = create_decoder_and_parse(args, input_file, false)?;
    if args.progressive && decoder.gainmap_present() {
        return Err(AvifError::UnknownError(
            "Progressive encoding is not supported with gain maps".into(),
        ));
    }
    if args.index.is_none() && decoder.image_count() > 1 {
        return Err(AvifError::UnknownError(
            "Re-encoding multi-frame AVIF sequences requires --index to select a frame".into(),
        ));
    }
    decoder.nth_image(args.index.unwrap_or(0))?;
    let src_image = decoder.image().ok_or(AvifError::NoContent)?;
    let mut image = match convert_image_format_and_depth(
        src_image,
        args.yuv_format,
        args.depth,
        args.cicp.as_ref(),
        args.ignore_icc || args.icc.is_some(),
        args.ignore_alpha,
        false,
    )? {
        Some(converted) => converted,
        None => src_image.try_deep_clone()?,
    };
    let mut gainmap = if decoder.gainmap_present() {
        Some(decoder.gainmap().try_deep_clone()?)
    } else {
        None
    };
    if args.ignore_icc {
        if let Some(gainmap) = &mut gainmap {
            gainmap.alt_icc.clear();
        }
    }
    apply_encode_overrides(args, &mut image, gainmap.as_mut())?;
    let encoder_arg_codec = match args.codec {
        CodecChoice::Aom => CodecChoice::Aom,
        #[cfg(feature = "avm")]
        CodecChoice::Avm => CodecChoice::Avm,
        #[cfg(feature = "jpegxl")]
        CodecChoice::Libjxl => CodecChoice::Libjxl,
        _ => CodecChoice::Auto,
    };
    let codec_choice = encoder_codec_choice_for_output(encoder_arg_codec, output_file);
    if args.progressive {
        let mut settings = create_encoder_settings(args, codec_choice);
        let mut encoder = Encoder::create_with_settings(&settings)?;
        settings.mutable.quality = 2.0;
        encoder.update_settings(&settings.mutable)?;
        encoder.add_image(&image)?;
        settings.mutable.quality = args.quality.unwrap_or(DEFAULT_ENCODE_QUALITY);
        encoder.update_settings(&settings.mutable)?;
        encoder.add_image(&image)?;
        return write_encoded_avif(encoder, output_file);
    }
    encode_single_image_to_avif(args, codec_choice, &image, gainmap.as_ref(), output_file)
}

#[cfg(not(feature = "encoder"))]
fn reencode(_args: &CommandLineArgs, _input_file: &String, _output_file: &str) -> AvifResult<()> {
    Err(AvifError::UnknownError(
        "AVIF output requires building with the encoder feature".into(),
    ))
}

#[cfg(feature = "encoder")]
fn encode(args: &CommandLineArgs, input_file: &str, output_file: &str) -> AvifResult<()> {
    let mut reader = <dyn Reader>::create(input_file)?;
    let reader_config = Config {
        yuv_format: args.yuv_format,
        depth: args.depth,
        ignore_icc: args.ignore_icc || args.icc.is_some(),
        ignore_exif: args.ignore_exif || args.exif.is_some(),
        ignore_xmp: args.ignore_xmp || args.xmp.is_some(),
        allow_sample_transform: args.allow_sample_transform,
        ..Default::default()
    };
    let (mut image, mut duration_ms, mut gainmap) = reader.read_frame(&reader_config)?;
    apply_read_ignore_flags(args, &mut image, gainmap.as_mut());
    apply_encode_overrides(args, &mut image, gainmap.as_mut())?;
    let has_more_frames = reader.has_more_frames();
    let codec_choice = encoder_codec_choice_for_output(args.codec, output_file);
    if !has_more_frames && !args.progressive {
        return encode_single_image_to_avif(
            args,
            codec_choice,
            &image,
            gainmap.as_ref(),
            output_file,
        );
    }
    let mut settings = create_encoder_settings(args, codec_choice);
    let mut encoder = Encoder::create_with_settings(&settings)?;
    if has_more_frames {
        if args.progressive {
            println!("Automatic progressive encoding can only have one input image.");
            return Err(AvifError::InvalidArgument);
        }
        if args.grid.is_some() {
            return Err(AvifError::UnknownError(
                "Grid encoding is not supported for image sequences.".into(),
            ));
        }
        loop {
            encoder.add_image_for_sequence(&image, duration_ms)?;
            if !reader.has_more_frames() {
                break;
            }
            (image, duration_ms, _) = reader.read_frame(&reader_config)?;
            apply_read_ignore_flags(args, &mut image, None);
        }
    } else {
        // Encode the base layer with very low quality.
        settings.mutable.quality = 2.0;
        encoder.update_settings(&settings.mutable)?;
        encoder.add_image(&image)?;
        // Encode the second layer with the requested quality.
        settings.mutable.quality = args.quality.unwrap_or(DEFAULT_ENCODE_QUALITY);
        encoder.update_settings(&settings.mutable)?;
        encoder.add_image(&image)?;
    }

    write_encoded_avif(encoder, output_file)
}

#[cfg(not(feature = "encoder"))]
fn encode(_args: &CommandLineArgs, _input_file: &str, _output_file: &str) -> AvifResult<()> {
    Err(AvifError::InvalidArgument)
}

fn can_decode(filename: &str) -> bool {
    match get_extension(filename).as_str() {
        "avif" => true,
        #[cfg(feature = "heic")]
        "heic" | "heif" => true,
        #[cfg(feature = "jpegxl")]
        "hxl" => true,
        _ => false,
    }
}

fn can_encode(filename: &str) -> bool {
    match get_extension(filename).as_str() {
        "avif" => true,
        #[cfg(feature = "jpegxl")]
        "hxl" => true,
        _ => false,
    }
}

fn validate_output_encoding_flags(args: &CommandLineArgs, output_filename: &str) -> AvifResult<()> {
    let extension = get_extension(output_filename);
    if let Some(quality) = args.quality {
        if extension != "jpg" && extension != "jpeg" && !can_encode(output_filename) {
            return Err(AvifError::UnknownError(
                "quality is only supported for jpeg or avif output".into(),
            ));
        }
        if !(0.0..=100.0).contains(&quality) {
            return Err(AvifError::UnknownError(
                "quality must be between 0 and 100 inclusive".into(),
            ));
        }
    }
    if args.depth.is_some() && extension != "png" && !can_encode(output_filename) {
        return Err(AvifError::UnknownError(
            "depth is only supported for png or avif output".into(),
        ));
    }
    if matches!(args.depth, Some(10 | 12)) && extension == "png" {
        return Err(AvifError::UnknownError(
            "depth 10 and 12 are not supported for png output".into(),
        ));
    }
    if args.depth == Some(16) && can_encode(output_filename) {
        return Err(AvifError::UnknownError(
            "depth 16 is not supported for avif output".into(),
        ));
    }
    if args.yuv_format.is_some() && !can_encode(output_filename) {
        return Err(AvifError::UnknownError(
            "yuv is only supported for avif output".into(),
        ));
    }
    if args.png_compress.is_some() && extension != "png" {
        return Err(AvifError::UnknownError(
            "png-compress-level is only supported for png output".into(),
        ));
    }
    if args.grid.is_some() && !can_encode(output_filename) {
        return Err(AvifError::UnknownError(
            "grid is only supported when encoding".into(),
        ));
    }
    Ok(())
}

fn validate_args(args: &CommandLineArgs) -> AvifResult<()> {
    if let Some(input_file) = &args.input_file {
        if args.tonemap {
            if args.info
                || args.extract_gainmap
                || args.extract_thumbnail
                || args.progressive
                || args.crop.is_some()
                || args.target_size.is_some()
            {
                return Err(AvifError::UnknownError(
                    "--tonemap cannot be combined with --info, --extract-gainmap, --extract-thumbnail, --progressive, --crop, or --target-size"
                        .into(),
                ));
            }
            if !args.headroom.is_finite() || args.headroom < 0.0 {
                return Err(AvifError::UnknownError(
                    "headroom must be finite and >= 0.0".into(),
                ));
            }
            let output_filename = args
                .output_file
                .as_ref()
                .ok_or_else(|| AvifError::UnknownError("output_file is required".into()))?;
            validate_output_encoding_flags(args, output_filename)?;
            return Ok(());
        }
        if can_decode(input_file) {
            if args.info {
                if args.output_file.is_some()
                    || args.quality.is_some()
                    || args.depth.is_some()
                    || args.index.is_some()
                    || args.extract_gainmap
                    || args.extract_thumbnail
                    || args.crop.is_some()
                    || args.target_size.is_some()
                    || args.grid.is_some()
                {
                    return Err(AvifError::UnknownError(
                        "--info contains unsupported extra arguments".into(),
                    ));
                }
            } else {
                let output_filename = args
                    .output_file
                    .as_deref()
                    .ok_or_else(|| AvifError::UnknownError("output_file is required".into()))?;
                validate_output_encoding_flags(args, output_filename)?;
                if args.crop.is_some() && args.progressive {
                    return Err(AvifError::UnknownError(
                        "crop is not supported with progressive decoding".into(),
                    ));
                }
                if args.target_size.is_some() && args.progressive {
                    return Err(AvifError::UnknownError(
                        "target-size is not supported with progressive decoding".into(),
                    ));
                }
                if args.grid.is_some() && args.progressive {
                    return Err(AvifError::UnknownError(
                        "grid is not supported with progressive encoding".into(),
                    ));
                }
            }
        } else {
            // TODO: b/403090413 - validate encoding args.
            if matches!(args.quality, Some(quality) if !(0.0..=100.0).contains(&quality)) {
                return Err(AvifError::UnknownError(
                    "quality must be between 0 and 100 inclusive".into(),
                ));
            }
            if args.target_size.is_some() {
                return Err(AvifError::UnknownError(
                    "target-size is only supported when decoding".into(),
                ));
            }
            if args.grid.is_some() && args.progressive {
                return Err(AvifError::UnknownError(
                    "grid is not supported with progressive encoding".into(),
                ));
            }
        }
    } else if !args.version {
        return Err(AvifError::UnknownError("input_file is required".into()));
    }
    Ok(())
}

fn binary_name() -> String {
    match std::env::args_os().next() {
        Some(binary_path) => {
            let path: &Path = binary_path.as_ref();
            match path.file_name() {
                Some(binary_name) => binary_name.to_string_lossy().into_owned(),
                None => "crabbyavif".into(),
            }
        }
        None => "crabbyavif".into(),
    }
}

pub fn main_impl<I, T>(itr: I) -> AvifResult<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let args = CommandLineArgs::parse_from(itr);
    if args.version {
        println!(
            "{} (crate {} {})",
            binary_name(),
            env!("CARGO_PKG_NAME"),    // Same as clap::crate_name!().
            env!("CARGO_PKG_VERSION"), // Same as clap::crate_version!().
        );
        println!("Available codecs:");
        println!("  {}", codec_versions());
    }
    validate_args(&args)?;
    if let Some(input_file) = &args.input_file {
        if args.tonemap {
            let output_file = args
                .output_file
                .as_deref()
                .ok_or_else(|| AvifError::UnknownError("output_file is required".into()))?;
            tonemap(&args, input_file, output_file)?;
        } else if can_decode(input_file) {
            if args.info {
                info(&args, input_file)?;
            } else if args.output_file.as_ref().is_some_and(|out| can_encode(out))
                && !args.extract_gainmap
                && !args.extract_thumbnail
                && args.crop.is_none()
                && args.target_size.is_none()
            {
                reencode(&args, input_file, args.output_file.as_ref().unwrap())?;
            } else {
                decode(&args, input_file)?;
            }
        } else if let Some(output_file) = &args.output_file {
            if can_encode(output_file) {
                encode(&args, input_file, output_file)?;
            } else {
                return Err(AvifError::UnknownError(
                    "Input/output file extensions not supported".into(),
                ));
            }
        } else {
            return Err(AvifError::UnknownError(format!(
                "Input file extension not supported: {}",
                get_extension(input_file)
            )));
        }
    }
    Ok(())
}
