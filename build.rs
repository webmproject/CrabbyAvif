// Build the required native library bindings.

// Not everything in this file is used in all the configurations.
#![allow(dead_code, unused_variables)]

use std::env;
use std::path::Path;
use std::path::PathBuf;

fn path_buf(inputs: &[&str]) -> PathBuf {
    let path: PathBuf = inputs.iter().collect();
    path
}

fn add_native_library(
    library_name: &str,
    library_dir: &str,
    library_path: PathBuf,
    header_file: PathBuf,
    extra_include_dir: PathBuf,
    allowlist_items: &[&str],
    bindings_path: PathBuf,
) {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let third_party = PathBuf::from(&project_root).join("third_party");
    let abs_library_dir = PathBuf::from(&third_party).join(library_dir);
    let abs_object_dir = PathBuf::from(&abs_library_dir).join(library_path);
    let object_file = format!("lib{library_name}.a");
    let library_file = PathBuf::from(&abs_object_dir).join(object_file);
    if !Path::new(&library_file).exists() {
        panic!("{library_name} not found. Run third_party/{library_dir}.cmd.");
    }
    println!("cargo:rustc-link-search={}", abs_object_dir.display());
    println!("cargo:rustc-link-lib=static={library_name}");

    // Generate bindings.
    let abs_header_file = PathBuf::from(&abs_library_dir).join(header_file);
    let extra_includes = PathBuf::from(&abs_library_dir).join(extra_include_dir);
    let extra_includes_str = format!("-I{}", extra_includes.display());
    let mut bindings = bindgen::Builder::default()
        .header(abs_header_file.into_os_string().into_string().unwrap())
        .clang_arg(extra_includes_str)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate_comments(false)
        .raw_line("#![allow(warnings)]");
    for allowlist_item in allowlist_items {
        bindings = bindings.allowlist_item(allowlist_item);
    }
    let bindings = bindings
        .generate()
        .unwrap_or_else(|_| panic!("Unable to generate bindings for {library_dir}"));
    bindings
        .write_to_file(bindings_path.as_path())
        .unwrap_or_else(|_| panic!("Couldn't write bindings for {library_dir}"));
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let build_target = std::env::var("TARGET").unwrap();
    let build_dir = if build_target.contains("android") {
        if build_target.contains("x86_64") {
            "build.android/x86_64"
        } else if build_target.contains("x86") {
            "build.android/x86"
        } else if build_target.contains("aarch64") {
            "build.android/aarch64"
        } else if build_target.contains("arm") {
            "build.android/arm"
        } else {
            panic!("Unknown target_arch for android. Must be one of x86, x86_64, arm, aarch64.");
        }
    } else {
        "build"
    };

    #[cfg(feature = "libgav1")]
    {
        add_native_library(
            "gav1",
            "libgav1",
            path_buf(&[build_dir]),
            path_buf(&["src", "gav1", "decoder.h"]),
            path_buf(&["src"]),
            &[
                "Libgav1DecoderCreate",
                "Libgav1DecoderDequeueFrame",
                "Libgav1DecoderDestroy",
                "Libgav1DecoderEnqueueFrame",
                "Libgav1DecoderSettingsInitDefault",
            ],
            path_buf(&["src", "codecs", "bindings", "libgav1.rs"]),
        );
        // libgav1 needs libstdc++ on *nix and libc++ on mac. TODO: what about windows?
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }

    #[cfg(feature = "libyuv")]
    add_native_library(
        "yuv",
        "libyuv",
        path_buf(&[build_dir]),
        path_buf(&["include", "libyuv.h"]),
        path_buf(&["include"]),
        &[
            "YuvConstants",
            "FilterMode",
            "ARGBAttenuate",
            "ARGBUnattenuate",
            "Convert16To8Plane",
            "HalfFloatPlane",
            "ScalePlane_12",
            "ScalePlane",
            "FilterMode_kFilterBilinear",
            "FilterMode_kFilterBox",
            "FilterMode_kFilterNone",
            "I010AlphaToARGBMatrixFilter",
            "I010AlphaToARGBMatrix",
            "I010ToARGBMatrixFilter",
            "I010ToARGBMatrix",
            "I012ToARGBMatrix",
            "I210AlphaToARGBMatrixFilter",
            "I210AlphaToARGBMatrix",
            "I210ToARGBMatrixFilter",
            "I210ToARGBMatrix",
            "I400ToARGBMatrix",
            "I410AlphaToARGBMatrix",
            "I410ToARGBMatrix",
            "I420AlphaToARGBMatrixFilter",
            "I420AlphaToARGBMatrix",
            "I420ToARGBMatrixFilter",
            "I420ToARGBMatrix",
            "I420ToRGB24MatrixFilter",
            "I420ToRGB24Matrix",
            "I420ToRGB565Matrix",
            "I420ToRGBAMatrix",
            "I422AlphaToARGBMatrixFilter",
            "I422AlphaToARGBMatrix",
            "I422ToARGBMatrixFilter",
            "I422ToARGBMatrix",
            "I422ToRGB24MatrixFilter",
            "I422ToRGB565Matrix",
            "I422ToRGBAMatrix",
            "I444AlphaToARGBMatrix",
            "I444ToARGBMatrix",
            "I444ToRGB24Matrix",
            "kYuv2020Constants",
            "kYuvF709Constants",
            "kYuvH709Constants",
            "kYuvI601Constants",
            "kYuvJPEGConstants",
            "kYuvV2020Constants",
            "kYvu2020Constants",
            "kYvuF709Constants",
            "kYvuH709Constants",
            "kYvuI601Constants",
            "kYvuJPEGConstants",
            "kYvuV2020Constants",
        ],
        path_buf(&["src", "reformat", "bindings", "libyuv.rs"]),
    );
}
