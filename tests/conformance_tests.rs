use crabby_avif::image::*;
use crabby_avif::utils::y4m;
use crabby_avif::*;

use std::fs::remove_file;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::process::Command;
use tempfile::NamedTempFile;

const TEST_DATA_PATH: &str = "/home2/files/avif/av1-avif/testFiles";
//const AVIFDEC_PATH: &str = "/opt/homebrew/bin/avifdec";
const AVIFDEC_PATH: &str = "/usr/local/google/home/vigneshv/code/libavif/build/avifdec";

#[derive(Copy, Clone)]
struct ExpectedImageInfo<'a> {
    filename: &'a str,
    width: u32,
    height: u32,
    depth: u8,
    yuv_format: PixelFormat,
    alpha_present: bool,
    full_range: bool,
    #[allow(unused)]
    color_primaries: u16,
    #[allow(unused)]
    transfer_characteristics: u16,
    #[allow(unused)]
    matrix_coefficients: u16,
}

fn verify_info(expected_info: &ExpectedImageInfo, image: &Image) {
    assert_eq!(image.width, expected_info.width);
    assert_eq!(image.height, expected_info.height);
    assert_eq!(image.depth, expected_info.depth);
    assert_eq!(image.yuv_format, expected_info.yuv_format);
    assert_eq!(image.alpha_present, expected_info.alpha_present);
    assert_eq!(image.full_range, expected_info.full_range);
    assert_eq!(image.color_primaries, expected_info.color_primaries.into());
    assert_eq!(
        image.transfer_characteristics,
        expected_info.transfer_characteristics.into()
    );
    assert_eq!(
        image.matrix_coefficients,
        expected_info.matrix_coefficients.into()
    );
}

fn get_tempfile() -> String {
    let file = NamedTempFile::new().expect("unable to open tempfile");
    let path = file.into_temp_path();
    let filename = String::from(path.to_str().unwrap());
    let _ = path.close();
    println!("### filename: {:#?}", filename);
    filename
}

fn write_y4m(image: &Image) -> String {
    let filename = get_tempfile();
    let mut y4m = y4m::Y4MWriter::create(&filename);
    assert!(y4m.write_frame(image));
    filename
}

fn run_avifdec(filename: &String) -> String {
    let mut outfile = get_tempfile();
    outfile.push_str(".y4m");
    let avifdec = Command::new(AVIFDEC_PATH)
        .arg("--no-strict")
        .arg("--jobs")
        .arg("8")
        .arg(filename)
        .arg(&outfile)
        .output()
        .unwrap();
    println!("avifdec: {:#?}", avifdec);
    assert!(avifdec.status.success());
    outfile
}

fn compare_files(file1: &String, file2: &String) -> bool {
    let f1 = File::open(file1).unwrap();
    let f2 = File::open(file2).unwrap();
    if f1.metadata().unwrap().len() != f2.metadata().unwrap().len() {
        return false;
    }
    let f1 = BufReader::new(f1);
    let f2 = BufReader::new(f2);
    for (byte1, byte2) in f1.bytes().zip(f2.bytes()) {
        if byte1.unwrap() != byte2.unwrap() {
            return false;
        }
    }
    true
}

fn decode_and_verify(expected_info: &ExpectedImageInfo) {
    let filename = String::from(format!("{TEST_DATA_PATH}/{}", expected_info.filename));
    let mut decoder = decoder::Decoder::default();
    decoder.settings.strictness = decoder::Strictness::None;
    let _ = decoder.set_io_file(&filename).expect("Failed to set IO");
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();
    verify_info(expected_info, &image);
    let res = decoder.next_image();
    assert!(res.is_ok());
    let image = res.unwrap();
    // Link-U 422 files have wrong subsampling in the Avif header(decoded one
    // is right).
    if !filename.contains("Link-U") || !filename.contains("yuv422") {
        verify_info(expected_info, &image);
    }

    // Write y4m.
    let y4m_file = write_y4m(image);
    // Write y4m by invoking avifdec.
    let gold_y4m_file = run_avifdec(&filename);
    // Compare.
    assert!(compare_files(&y4m_file, &gold_y4m_file));
    let _ = remove_file(y4m_file);
    let _ = remove_file(gold_y4m_file);
    println!("filename: {filename}");
}

// If more files are added to this array, update the call to generate_tests macro below.
const EXPECTED_INFOS: [ExpectedImageInfo; 172] = [
    // index: 0
    ExpectedImageInfo {
        filename: "Apple/edge_case_testing/non_compliant/truncated_elementary_stream.avif",
        width: 1024,
        height: 768,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 1
    ExpectedImageInfo {
        filename: "Apple/edge_case_testing/unknown_properties/free_property.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 2
    ExpectedImageInfo {
        filename: "Apple/edge_case_testing/unknown_properties/unknown_nonessential_property.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 3
    ExpectedImageInfo {
        filename: "Apple/multilayer_examples/animals_00_multilayer_a1lx.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 4
    ExpectedImageInfo {
        filename: "Apple/multilayer_examples/animals_00_multilayer_a1op.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 5
    ExpectedImageInfo {
        filename: "Apple/multilayer_examples/animals_00_multilayer_a1op_lsel.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 6
    ExpectedImageInfo {
        filename: "Apple/multilayer_examples/animals_00_multilayer_grid_a1lx.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 7
    ExpectedImageInfo {
        filename: "Apple/multilayer_examples/animals_00_multilayer_grid_lsel.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 8
    ExpectedImageInfo {
        filename: "Apple/multilayer_examples/animals_00_multilayer_lsel.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 9
    ExpectedImageInfo {
        filename: "Apple/multilayer_examples/animals_00_singlelayer.avif",
        width: 2048,
        height: 1536,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 10
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.avif",
        width: 1204,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 11
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.monochrome.avif",
        width: 1204,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 12
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.monochrome.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 13
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.monochrome.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 14
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.monochrome.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 15
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 16
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 17
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.10bpc.yuv420.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 18
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.avif",
        width: 1204,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 19
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.monochrome.avif",
        width: 1204,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 20
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.monochrome.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 21
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.monochrome.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 22
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.monochrome.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 23
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 24
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 25
    ExpectedImageInfo {
        filename: "Link-U/fox.profile0.8bpc.yuv420.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 26
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.10bpc.yuv444.avif",
        width: 1204,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 27
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.10bpc.yuv444.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 28
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.10bpc.yuv444.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 29
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.10bpc.yuv444.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 30
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.8bpc.yuv444.avif",
        width: 1204,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 31
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.8bpc.yuv444.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 32
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.8bpc.yuv444.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 33
    ExpectedImageInfo {
        filename: "Link-U/fox.profile1.8bpc.yuv444.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 34
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.avif",
        width: 1204,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 35
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.monochrome.avif",
        width: 1204,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 36
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.monochrome.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 37
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.monochrome.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 38
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.monochrome.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 39
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 40
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 41
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.10bpc.yuv422.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 42
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.avif",
        width: 1204,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 43
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.monochrome.avif",
        width: 1204,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 44
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.monochrome.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 45
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.monochrome.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 46
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.monochrome.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 47
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 48
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 49
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv420.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 50
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.avif",
        width: 1204,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 51
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.monochrome.avif",
        width: 1204,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 52
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.monochrome.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 53
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.monochrome.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 54
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.monochrome.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 55
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 56
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 57
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv422.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 58
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.avif",
        width: 1204,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 59
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.monochrome.avif",
        width: 1204,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 60
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.monochrome.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 61
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.monochrome.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 62
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.monochrome.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 63
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 64
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 65
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.12bpc.yuv444.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 66
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.avif",
        width: 1204,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 67
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.monochrome.avif",
        width: 1204,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 68
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.monochrome.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 69
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.monochrome.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 70
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.monochrome.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 71
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.odd-height.avif",
        width: 1204,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 72
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.odd-width.avif",
        width: 1203,
        height: 800,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 73
    ExpectedImageInfo {
        filename: "Link-U/fox.profile2.8bpc.yuv422.odd-width.odd-height.avif",
        width: 1203,
        height: 799,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 74
    ExpectedImageInfo {
        filename: "Link-U/hato.profile0.10bpc.yuv420.monochrome.no-cdef.no-restoration.avif",
        width: 3082,
        height: 2048,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 75
    ExpectedImageInfo {
        filename: "Link-U/hato.profile0.10bpc.yuv420.no-cdef.no-restoration.avif",
        width: 3082,
        height: 2048,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 76
    ExpectedImageInfo {
        filename: "Link-U/hato.profile0.8bpc.yuv420.monochrome.no-cdef.avif",
        width: 3082,
        height: 2048,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 77
    ExpectedImageInfo {
        filename: "Link-U/hato.profile0.8bpc.yuv420.no-cdef.avif",
        width: 3082,
        height: 2048,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 78
    ExpectedImageInfo {
        filename: "Link-U/hato.profile2.10bpc.yuv422.monochrome.no-cdef.no-restoration.avif",
        width: 3082,
        height: 2048,
        depth: 10,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 79
    ExpectedImageInfo {
        filename: "Link-U/hato.profile2.10bpc.yuv422.no-cdef.no-restoration.avif",
        width: 3082,
        height: 2048,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 80
    ExpectedImageInfo {
        filename: "Link-U/hato.profile2.12bpc.yuv422.monochrome.avif",
        width: 3082,
        height: 2048,
        depth: 12,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 2,
        matrix_coefficients: 9,
    },
    // index: 81
    ExpectedImageInfo {
        filename: "Link-U/hato.profile2.12bpc.yuv422.monochrome.no-cdef.no-restoration.avif",
        width: 3082,
        height: 2048,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 82
    ExpectedImageInfo {
        filename: "Link-U/hato.profile2.12bpc.yuv422.no-cdef.no-restoration.avif",
        width: 3082,
        height: 2048,
        depth: 12,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 83
    ExpectedImageInfo {
        filename: "Link-U/hato.profile2.8bpc.yuv422.monochrome.no-cdef.avif",
        width: 3082,
        height: 2048,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 84
    ExpectedImageInfo {
        filename: "Link-U/hato.profile2.8bpc.yuv422.no-cdef.avif",
        width: 3082,
        height: 2048,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 85
    ExpectedImageInfo {
        filename: "Link-U/kimono.avif",
        width: 722,
        height: 1024,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 86
    ExpectedImageInfo {
        filename: "Link-U/kimono.crop.avif",
        width: 722,
        height: 1024,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 87
    ExpectedImageInfo {
        filename: "Link-U/kimono.mirror-horizontal.avif",
        width: 722,
        height: 1024,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 88
    ExpectedImageInfo {
        filename: "Link-U/kimono.mirror-vertical.avif",
        width: 722,
        height: 1024,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 89
    ExpectedImageInfo {
        filename: "Link-U/kimono.mirror-vertical.rotate270.avif",
        width: 1024,
        height: 722,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 90
    ExpectedImageInfo {
        filename: "Link-U/kimono.mirror-vertical.rotate270.crop.avif",
        width: 1024,
        height: 722,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 91
    ExpectedImageInfo {
        filename: "Link-U/kimono.rotate270.avif",
        width: 1024,
        height: 722,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 92
    ExpectedImageInfo {
        filename: "Link-U/kimono.rotate90.avif",
        width: 1024,
        height: 722,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 9,
    },
    // index: 93
    ExpectedImageInfo {
        filename: "Microsoft/Chimera_10bit_cropped_to_1920x1008.avif",
        width: 1920,
        height: 1080,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 94
    ExpectedImageInfo {
        filename: "Microsoft/Chimera_10bit_cropped_to_1920x1008_with_HDR_metadata.avif",
        width: 1920,
        height: 1080,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 10,
    },
    // index: 95
    ExpectedImageInfo {
        filename: "Microsoft/Chimera_8bit_cropped_480x256.avif",
        width: 480,
        height: 270,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 96
    ExpectedImageInfo {
        filename: "Microsoft/Irvine_CA.avif",
        width: 480,
        height: 640,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 97
    ExpectedImageInfo {
        filename: "Microsoft/Mexico.avif",
        width: 1920,
        height: 1080,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 98
    ExpectedImageInfo {
        filename: "Microsoft/Mexico_YUV444.avif",
        width: 960,
        height: 540,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 99
    ExpectedImageInfo {
        filename: "Microsoft/Monochrome.avif",
        width: 1280,
        height: 720,
        depth: 8,
        yuv_format: PixelFormat::Monochrome,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 1,
        matrix_coefficients: 1,
    },
    // index: 100
    ExpectedImageInfo {
        filename: "Microsoft/Ronda_rotate90.avif",
        width: 1920,
        height: 1080,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 101
    ExpectedImageInfo {
        filename: "Microsoft/Summer_Nature_4k.avif",
        width: 3840,
        height: 2160,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 102
    ExpectedImageInfo {
        filename: "Microsoft/Summer_in_Tomsk_720p_5x4_grid.avif",
        width: 6400,
        height: 2880,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 103
    ExpectedImageInfo {
        filename: "Microsoft/Tomsk_with_thumbnails.avif",
        width: 1280,
        height: 720,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 1,
    },
    // index: 104
    ExpectedImageInfo {
        filename: "Microsoft/bbb_4k.avif",
        width: 3840,
        height: 2160,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 105
    ExpectedImageInfo {
        filename: "Microsoft/bbb_alpha_inverted.avif",
        width: 3840,
        height: 2160,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: true,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 1,
    },
    // index: 106
    ExpectedImageInfo {
        filename: "Microsoft/kids_720p.avif",
        width: 1280,
        height: 720,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: true,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 107
    ExpectedImageInfo {
        filename: "Microsoft/reduced_still_picture_header.avif",
        width: 1280,
        height: 720,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 1,
    },
    // index: 108
    ExpectedImageInfo {
        filename: "Microsoft/still_picture.avif",
        width: 1280,
        height: 720,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 1,
    },
    // index: 109
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01000_cicp9-16-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 0,
    },
    // index: 110
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01000_cicp9-16-9_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 111
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01000_cicp9-16-9_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 112
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01000_cicp9-16-9_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 113
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01000_cicp9-16-9_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 114
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01000_cicp9-16-9_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 115
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01000_cicp9-16-9_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 116
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01650_cicp9-16-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 0,
    },
    // index: 117
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01650_cicp9-16-9_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 118
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01650_cicp9-16-9_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 119
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01650_cicp9-16-9_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 120
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01650_cicp9-16-9_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 121
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01650_cicp9-16-9_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 122
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos01650_cicp9-16-9_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 123
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos07296_cicp9-16-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 0,
    },
    // index: 124
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos07296_cicp9-16-9_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 125
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos07296_cicp9-16-9_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 126
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos07296_cicp9-16-9_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 127
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos07296_cicp9-16-9_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 128
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos07296_cicp9-16-9_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 129
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos07296_cicp9-16-9_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 130
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos12920_cicp9-16-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 0,
    },
    // index: 131
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos12920_cicp9-16-9_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 132
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos12920_cicp9-16-9_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 133
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos12920_cicp9-16-9_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 134
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos12920_cicp9-16-9_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 135
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos12920_cicp9-16-9_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 136
    ExpectedImageInfo {
        filename: "Netflix/avif/hdr_cosmos12920_cicp9-16-9_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 10,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
    },
    // index: 137
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01000_cicp1-13-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
    },
    // index: 138
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01000_cicp1-13-6_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 139
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01000_cicp1-13-6_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 140
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01000_cicp1-13-6_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 141
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01000_cicp1-13-6_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 142
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01000_cicp1-13-6_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 143
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01000_cicp1-13-6_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 144
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01650_cicp1-13-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
    },
    // index: 145
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01650_cicp1-13-6_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 146
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01650_cicp1-13-6_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 147
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01650_cicp1-13-6_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 148
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01650_cicp1-13-6_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 149
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01650_cicp1-13-6_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 150
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos01650_cicp1-13-6_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 151
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos07296_cicp1-13-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
    },
    // index: 152
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos07296_cicp1-13-6_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 153
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos07296_cicp1-13-6_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 154
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos07296_cicp1-13-6_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 155
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos07296_cicp1-13-6_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 156
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos07296_cicp1-13-6_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 157
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos07296_cicp1-13-6_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 158
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos12920_cicp1-13-0_lossless.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
    },
    // index: 159
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos12920_cicp1-13-6_yuv420_limited_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 160
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos12920_cicp1-13-6_yuv420_limited_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 161
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos12920_cicp1-13-6_yuv420_limited_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 162
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos12920_cicp1-13-6_yuv444_full_qp10.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 163
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos12920_cicp1-13-6_yuv444_full_qp20.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 164
    ExpectedImageInfo {
        filename: "Netflix/avif/sdr_cosmos12920_cicp1-13-6_yuv444_full_qp40.avif",
        width: 2048,
        height: 858,
        depth: 8,
        yuv_format: PixelFormat::Yuv444,
        alpha_present: false,
        full_range: true,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
    },
    // index: 165
    ExpectedImageInfo {
        filename: "Netflix/avis/Chimera-AV1-10bit-480x270.avif",
        width: 480,
        height: 270,
        depth: 10,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 166
    ExpectedImageInfo {
        filename: "Netflix/avis/alpha_video.avif",
        width: 640,
        height: 480,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: true,
        full_range: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 1,
    },
    // index: 167
    ExpectedImageInfo {
        filename: "Xiph/abandoned_filmgrain.avif",
        width: 1404,
        height: 936,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 168
    ExpectedImageInfo {
        filename: "Xiph/fruits_2layer_thumbsize.avif",
        width: 1296,
        height: 864,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 169
    ExpectedImageInfo {
        filename: "Xiph/quebec_3layer_op2.avif",
        width: 360,
        height: 182,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 170
    ExpectedImageInfo {
        filename: "Xiph/tiger_3layer_1res.avif",
        width: 1216,
        height: 832,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
    // index: 171
    ExpectedImageInfo {
        filename: "Xiph/tiger_3layer_3res.avif",
        width: 1216,
        height: 832,
        depth: 8,
        yuv_format: PixelFormat::Yuv420,
        alpha_present: false,
        full_range: false,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
    },
];

macro_rules! generate_tests {
    ($from: expr, $to: expr) => {
        seq_macro::seq!(N in $from..$to {
            #(#[test_case::test_case(N)])*
            fn test_conformance(index: usize) {
                decode_and_verify(&EXPECTED_INFOS[index]);
            }
        });
    };
}

generate_tests!(0, 172);
//generate_tests!(102, 103);
