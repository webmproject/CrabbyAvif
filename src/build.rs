use std::env;
use std::path::Path;

fn find_library(library: &str, local_sub_path: &str) {
    if cfg!(target_os = "android") {
        // android arm64
        println!(
            "cargo:rustc-link-search=/Users/vigneshv/code/libavif/ext/dav1d/build/arm64-v8a/src"
        );
        println!("cargo:rustc-link-lib=static=dav1d");
        return;
    }
    // Check if a locally built static version of the library is available.
    let project_root = env!("CARGO_MANIFEST_DIR");
    let local_library_dir = format!("{project_root}/third_party/{local_sub_path}");
    let local_library = format!("{local_library_dir}/lib{library}.a");
    if Path::new(&local_library).exists() {
        println!("cargo:rustc-link-search={local_library_dir}");
        println!("cargo:rustc-link-lib=static={library}");
        return;
    }
    // No locally built library was found. Use the system library (if it exists). If there is no
    // system library available, this will result in a linker error.
    println!("cargo:rustc-link-lib={library}");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "dav1d")]
    find_library("dav1d", "dav1d/build/src");

    #[cfg(feature = "libgav1")]
    {
        find_library("gav1", "libgav1/build");
        // libgav1 needs libstdc++ on *nix and libc++ on mac..
        // TODO: what about windows?
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }
}
