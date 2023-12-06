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

    #[cfg(feature = "dav1d")]
    add_native_library(
        "dav1d",
        "dav1d",
        path_buf(&[build_dir, "src"]),
        path_buf(&["include", "dav1d", "dav1d.h"]),
        path_buf(&[build_dir, "include", "dav1d"]),
        &[
            "EAGAIN",
            "dav1d_close",
            "dav1d_data_unref",
            "dav1d_data_wrap",
            "dav1d_default_settings",
            "dav1d_error",
            "dav1d_get_picture",
            "dav1d_open",
            "dav1d_picture_unref",
            "dav1d_send_data",
        ],
    );

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
        );
        // libgav1 needs libstdc++ on *nix and libc++ on mac. TODO: what about windows?
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }
}
