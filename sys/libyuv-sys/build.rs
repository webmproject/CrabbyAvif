// Build rust library and bindings for libyuv.

use std::env;
use std::path::Path;
use std::path::PathBuf;

fn path_buf(inputs: &[&str]) -> PathBuf {
    let path: PathBuf = inputs.iter().collect();
    path
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

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let abs_library_dir = PathBuf::from(&project_root).join("libyuv");
    let abs_object_dir = PathBuf::from(&abs_library_dir).join(build_dir);
    let object_file = format!("libyuv.a");
    let library_file = PathBuf::from(&abs_object_dir).join(object_file);
    if !Path::new(&library_file).exists() {
        panic!("libyuv not found. Run libyuv.cmd.");
    }
    println!("cargo:rustc-link-search={}", abs_object_dir.display());
    println!("cargo:rustc-link-lib=static=yuv");

    // Generate bindings.
    let header_file = PathBuf::from(&abs_library_dir).join(path_buf(&["include", "libyuv.h"]));
    let version_dir = PathBuf::from(&abs_library_dir).join(path_buf(&["include"]));
    let outfile = PathBuf::from(&project_root).join(path_buf(&["src", "libyuv.rs"]));
    let extra_includes_str = format!("-I{}", version_dir.display());
    let mut bindings = bindgen::Builder::default()
        .header(header_file.into_os_string().into_string().unwrap())
        .clang_arg(extra_includes_str)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate_comments(false);
    let allowlist_items = &[
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
    ];
    for allowlist_item in allowlist_items {
        bindings = bindings.allowlist_item(allowlist_item);
    }
    let bindings = bindings
        .generate()
        .unwrap_or_else(|_| panic!("Unable to generate bindings for libyuv."));
    bindings
        .write_to_file(outfile.as_path())
        .unwrap_or_else(|_| panic!("Couldn't write bindings for libyuv"));
    println!(
        "cargo:rustc-env=CRABBYAVIF_LIBYUV_BINDINGS_RS={}",
        outfile.display()
    );
}
