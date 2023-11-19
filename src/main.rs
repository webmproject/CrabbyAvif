use std::env;
use std::process::Command;

use rust_libavif::decoder::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("Usage: {} <input_avif> <output> [--no-png]", args[0]);
        std::process::exit(1);
    }

    let settings = AvifDecoderSettings {
        source: AvifDecoderSource::Auto,
        ignore_exif: false,
        ignore_icc: false,
    };
    let mut decoder: AvifDecoder = Default::default();
    decoder.settings = settings;
    decoder.set_file(&args[1]);
    let image = decoder.parse();
    println!("image after parse: {:#?}", image);
    if image.is_none() {
        println!("parse failed!");
        std::process::exit(1);
    }

    println!("\n^^^ decoder public properties ^^^");
    println!("image_count: {}", decoder.image_count);
    println!("timescale: {}", decoder.timescale);
    println!("duration_in_timescales: {}", decoder.duration_in_timescales);
    println!("duration: {}", decoder.duration);
    println!("repetition_count: {}", decoder.repetition_count);
    println!("$$$ end decoder public properties $$$\n");

    let image_count = decoder.image_count;
    //let image_count = 1;
    let mut y4m: rust_libavif::utils::Y4MWriter = Default::default();
    y4m.filename = args[2].clone();

    for _i in 0..image_count {
        let image = decoder.next_image();
        println!("image after decode: {:#?}", image);
        if image.is_none() {
            println!("next_image failed!");
            std::process::exit(1);
        }

        let ret = y4m.write_frame(image.unwrap());
        if !ret {
            println!("error writing y4m file");
            std::process::exit(1);
        }
    }
    println!("wrote {} frames into {}", image_count, args[2]);
    if args.len() == 3 {
        if image_count <= 1 {
            let ffmpeg_infile = format!("{}", args[2]);
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
            let ffmpeg_infile = format!("{}", args[2]);
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
