use std::env;
use std::process::Command;

use rust_libavif::decoder::*;
use rust_libavif::*;

fn main() {
    // let data: [u8; 32] = [
    //     0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x61, 0x76, 0x69, 0x66, 0x00, 0x00, 0x00,
    //     0x00, 0x61, 0x76, 0x69, 0x66, 0x6d, 0x69, 0x66, 0x31, 0x6d, 0x69, 0x61, 0x66, 0x4d, 0x41,
    //     0x31, 0x41,
    // ];
    // let data: [u8; 32] = [
    //     0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x61, 0x76, 0x69, 0x67, 0x00, 0x00, 0x00,
    //     0x00, 0x61, 0x76, 0x69, 0x68, 0x6d, 0x69, 0x66, 0x31, 0x6d, 0x69, 0x61, 0x66, 0x4d, 0x41,
    //     0x31, 0x41,
    // ];
    // let val = AvifDecoder::peek_compatible_file_type(&data);
    // println!("val: {val}");
    // return;

    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("Usage: {} <input_avif> <output> [--no-png]", args[0]);
        std::process::exit(1);
    }
    let image_count;
    {
        let settings = AvifDecoderSettings {
            source: AvifDecoderSource::Auto,
            ignore_exif: false,
            ignore_icc: false,
            strictness: AvifStrictness::None,
            allow_progressive: false,
        };
        let mut decoder: AvifDecoder = Default::default();
        decoder.settings = settings;
        match decoder.set_io_file(&args[1]) {
            Ok(_) => {}
            Err(err) => {
                println!("failed to set file io: {:#?}", err);
                std::process::exit(1);
            }
        };
        let image = match decoder.parse() {
            Ok(x) => x,
            Err(err) => {
                println!("decoder.parse failed: {:#?}", err);
                std::process::exit(1);
            }
        };
        println!("image after parse: {:#?}", image);

        println!("\n^^^ decoder public properties ^^^");
        println!("image_count: {}", decoder.image_count);
        println!("timescale: {}", decoder.timescale);
        println!("duration_in_timescales: {}", decoder.duration_in_timescales);
        println!("duration: {}", decoder.duration);
        println!("repetition_count: {}", decoder.repetition_count);
        println!("$$$ end decoder public properties $$$\n");

        image_count = decoder.image_count;
        //image_count = 1;
        let mut y4m: rust_libavif::utils::Y4MWriter = Default::default();
        y4m.filename = Some(args[2].clone());

        for _i in 0..image_count {
            let image = decoder.next_image();
            println!("image after decode: {:#?}", image);
            if image.is_err() {
                println!("next_image failed! {:#?}", image);
                std::process::exit(1);
            }

            let ret = y4m.write_frame(image.unwrap());
            if !ret {
                println!("error writing y4m file");
                std::process::exit(1);
            }
        }
        println!("wrote {} frames into {}", image_count, args[2]);
    }
    if args.len() == 3 {
        if image_count <= 1 {
            let ffmpeg_infile = args[2].to_string();
            let ffmpeg_outfile = format!("{}.png", args[2]);
            let ffmpeg = Command::new("ffmpeg")
                .arg("-i")
                .arg(ffmpeg_infile)
                .arg("-frames:v")
                .arg("1")
                .arg("-y")
                .arg(ffmpeg_outfile)
                .output()
                .unwrap();
            if !ffmpeg.status.success() {
                println!("ffmpeg to convert to png failed");
                std::process::exit(1);
            }
            println!("wrote {}.png", args[2]);
        } else {
            let ffmpeg_infile = args[2].to_string();
            let ffmpeg_outfile = format!("{}.gif", args[2]);
            let ffmpeg = Command::new("ffmpeg")
                .arg("-i")
                .arg(ffmpeg_infile)
                .arg("-y")
                .arg(ffmpeg_outfile)
                .output()
                .unwrap();
            if !ffmpeg.status.success() {
                println!("ffmpeg to convert to gif failed");
                std::process::exit(1);
            }
            println!("wrote {}.gif", args[2]);
        }
    }
    std::process::exit(0);
}
