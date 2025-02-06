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

use clap::Parser;

use crabby_avif::decoder::track::RepetitionCount;
use crabby_avif::decoder::*;
use crabby_avif::*;

mod writer;

use writer::jpeg::JpegWriter;
use writer::png::PngWriter;
use writer::y4m::Y4MWriter;
use writer::Writer;

use std::fs::File;

#[derive(Parser)]
struct CommandLineArgs {
    /// Disable strict decoding, which disables strict validation checks and errors
    #[arg(long)]
    no_strict: Option<bool>,

    /// Decode all frames and display all image information instead of saving to disk
    #[arg(short, long, default_value = "false")]
    info: bool,

    /// Input AVIF file
    #[arg(allow_hyphen_values = false)]
    input_file: String,

    /// Output file
    #[arg(allow_hyphen_values = false)]
    output_file: Option<String>,

    #[arg(long)]
    jobs: Option<u32>,

    /// When decoding an image sequence or progressive image, specify which frame index to decode
    /// (Default: 0)
    #[arg(long, short = 'I')]
    index: Option<u32>,

    /// Output depth, either 8 or 16. (PNG only; For y4m/yuv, source depth is retained; JPEG is
    /// always 8bit)
    #[arg(long, short = 'd')]
    depth: Option<u8>,

    /// Output quality in 0..100. (JPEG only, default: 90)
    #[arg(long, short = 'q')]
    quality: Option<u8>,
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
        format!("Absent")
    } else {
        format!("Present ({} bytes)", data.len())
    }
}

fn print_image_info(decoder: &Decoder) {
    let image = decoder.image().unwrap();
    let mut image_data = vec![
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
            format!(
                "{}",
                match (image.alpha_present, image.alpha_premultiplied) {
                    (true, true) => "Premultiplied",
                    (true, false) => "Not premultiplied",
                    (false, _) => "Absent",
                }
            ),
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
        image_data.push((0, "Transformations", format!("None")));
    } else {
        image_data.push((0, "Transformations", format!("")));
        if let Some(pasp) = image.pasp {
            image_data.push((
                1,
                "pasp (Aspect Ratio)",
                format!("{}/{}", pasp.h_spacing, pasp.v_spacing),
            ));
        }
        if let Some(_clap) = image.clap {
            // TODO: b/394162563 - print clap info.
            image_data.push((1, "clap (Clean Aperture)", format!("")));
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
        let gainmap = decoder.gainmap();
        let gainmap_image = &gainmap.image;
        image_data.extend_from_slice(&[
            (
                0,
                "Gainmap",
                format!(
                "{}x{} pixels, {} bit, {:#?}, {:#?} Range, Matrix Coeffs. {:#?}, Base Image is {}",
                gainmap_image.width,
                gainmap_image.height,
                gainmap_image.depth,
                gainmap_image.yuv_format,
                gainmap_image.yuv_range,
                gainmap_image.matrix_coefficients,
                if gainmap.metadata.base_hdr_headroom.0 == 0 { "SDR" } else { "HDR" },
            ),
            ),
            (0, "Alternate image", format!("")),
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
            if let Some(clli) = gainmap_image.clli {
                (1, "CLLI", format!("{}, {}", clli.max_cll, clli.max_pall))
            } else {
                (1, "", "".into())
            },
        ])
    } else {
        // TODO: b/394162563 - check if we need to report the present but ignored case.
        image_data.push((0, "Gainmap", format!("Absent")));
    }
    if image.image_sequence_track_present {
        image_data.push((
            0,
            "Repeat Count",
            match decoder.repetition_count() {
                RepetitionCount::Finite(x) => format!("{x}"),
                RepetitionCount::Infinite => format!("Infinite"),
                RepetitionCount::Unknown => format!("Unknown"),
            },
        ));
    }
    print_data_as_columns(&image_data);
}

fn max_threads(jobs: &Option<u32>) -> u32 {
    match jobs {
        Some(x) => {
            if *x == 0 {
                // TODO: b/394162563 - Query the number of available CPU cores for this case.
                1
            } else {
                *x
            }
        }
        None => 1,
    }
}

fn create_decoder_and_parse(args: &CommandLineArgs) -> AvifResult<Decoder> {
    let settings = Settings {
        strictness: if args.no_strict.unwrap_or(false) {
            Strictness::None
        } else {
            Strictness::All
        },
        image_content_to_decode: ImageContentType::All,
        max_threads: max_threads(&args.jobs),
        ..Settings::default()
    };
    let mut decoder = Decoder::default();
    decoder.settings = settings;
    decoder
        .set_io_file(&args.input_file)
        .or(Err(AvifError::UnknownError(
            "Cannot open input file".into(),
        )))?;
    decoder.parse()?;
    Ok(decoder)
}

fn info(args: &CommandLineArgs) -> AvifResult<()> {
    if args.output_file.is_some() {
        return Err(AvifError::UnknownError(
            "output_file is not allowed with --info".into(),
        ));
    }
    let mut decoder = create_decoder_and_parse(&args)?;
    println!("Image decoded: {}", args.input_file);
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

fn get_extension(filename: &str) -> &str {
    std::path::Path::new(filename)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
}

fn decode(args: &CommandLineArgs) -> AvifResult<()> {
    if args.output_file.is_none() {
        return Err(AvifError::UnknownError("output_file is required".into()));
    }
    if let Some(depth) = args.depth {
        if depth != 8 && depth != 16 {
            return Err(AvifError::UnknownError(format!(
                "Invalid depth requested: {depth}"
            )));
        }
    }
    if let Some(quality) = args.quality {
        if quality > 100 {
            return Err(AvifError::UnknownError(format!(
                "Invalid output quality requested: {quality}"
            )));
        }
    }
    let max_threads = max_threads(&args.jobs);
    println!(
        "Decoding with {max_threads} worker thread{}, please wait...",
        if max_threads == 1 { "" } else { "s" }
    );
    let mut decoder = create_decoder_and_parse(&args)?;
    decoder.nth_image(args.index.unwrap_or(0))?;
    println!("Image Decoded: {}", args.input_file);
    println!("Image details:");
    print_image_info(&decoder);

    let output_filename = &args.output_file.as_ref().unwrap().as_str();
    let image = decoder.image().unwrap();
    let extension = get_extension(output_filename);
    let mut writer: Box<dyn Writer> = match extension {
        "y4m" | "yuv" => {
            if !image.icc.is_empty() || !image.exif.is_empty() || !image.xmp.is_empty() {
                println!("Warning: metadata dropped when saving to {extension}");
            }
            Box::new(Y4MWriter::create(extension == "yuv"))
        }
        "png" => Box::new(PngWriter { depth: args.depth }),
        "jpg" | "jpeg" => Box::new(JpegWriter {
            quality: args.quality,
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
    writer.write_frame(&mut output_file, image)?;
    println!(
        "Wrote image at index {} to output {}",
        args.index.unwrap_or(0),
        output_filename,
    );
    Ok(())
}

fn main() {
    let args = CommandLineArgs::parse();
    let res = if args.info { info(&args) } else { decode(&args) };
    match res {
        Ok(_) => std::process::exit(0),
        Err(err) => {
            eprintln!("ERROR: {:#?}", err);
            std::process::exit(1);
        }
    }
}
