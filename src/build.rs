use std::env;
use std::path::Path;
use std::path::PathBuf;

use bindgen::CargoCallbacks;

fn add_native_library(
    library_name: &str,
    library_dir: &str,
    library_path: PathBuf,
    header_file: PathBuf,
    extra_include_dir: PathBuf,
) {
    if cfg!(target_os = "android") {
        // android arm64
        println!(
            "cargo:rustc-link-search=/Users/vigneshv/code/libavif/ext/dav1d/build/arm64-v8a/src"
        );
        println!("cargo:rustc-link-lib=static=dav1d");
        return;
    }
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
    let bindings = bindgen::Builder::default()
        .header(abs_header_file.into_os_string().into_string().unwrap())
        .clang_arg(extra_includes_str)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .layout_tests(false)
        .raw_line("#![allow(warnings)]")
        .generate()
        .unwrap_or_else(|_| panic!("Unable to generate bindings for {library_dir}"));

    let rs_file = format!("{library_dir}.rs");
    let bindings_path = PathBuf::from(&project_root)
        .join("src")
        .join("codecs")
        .join("bindings")
        .join(rs_file);
    bindings
        .write_to_file(bindings_path.as_path())
        .unwrap_or_else(|_| panic!("Couldn't write bindings for {library_dir}"));
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "dav1d")]
    add_native_library(
        "dav1d",
        "dav1d",
        PathBuf::from("build/src"),
        PathBuf::from("include/dav1d/dav1d.h"),
        PathBuf::from("build/include/dav1d"),
    );

    #[cfg(feature = "libgav1")]
    {
        add_native_library(
            "gav1",
            "libgav1",
            PathBuf::from("build"),
            PathBuf::from("src/gav1/decoder.h"),
            PathBuf::from("src"),
        );
        // libgav1 needs libstdc++ on *nix and libc++ on mac. TODO: what about windows?
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }
}
