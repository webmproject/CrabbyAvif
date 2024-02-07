// Build rust library and bindings for dav1d.

use std::env;
use std::path::PathBuf;

fn path_buf(inputs: &[&str]) -> PathBuf {
    let path: PathBuf = inputs.iter().collect();
    path
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let build_target = std::env::var("TARGET").unwrap();
    if !build_target.contains("android") {
        panic!("Not an android target: {build_target}");
    };

    // Generate bindings.
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let header_file = PathBuf::from(&project_root).join("wrapper.h");
    let outfile = PathBuf::from(&project_root).join(path_buf(&["src", "bindings.rs"]));
    let host_tag = "linux-x86_64"; // TODO: Support windows and mac.
    let sysroot = format!(
        "{}/toolchains/llvm/prebuilt/{}/sysroot/",
        env!("ANDROID_NDK_ROOT"),
        host_tag
    );
    let mut bindings = bindgen::Builder::default()
        .header(header_file.into_os_string().into_string().unwrap())
        .clang_arg(format!("--sysroot={sysroot}"))
        .clang_arg(format!("--target={build_target}"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate_comments(false);
    /*
    let allowlist_items = &[
        "dav1d_close",
        "dav1d_data_unref",
        "dav1d_data_wrap",
        "dav1d_default_settings",
        "dav1d_error",
        "dav1d_get_picture",
        "dav1d_open",
        "dav1d_picture_unref",
        "dav1d_send_data",
    ];
    for allowlist_item in allowlist_items {
        bindings = bindings.allowlist_item(allowlist_item);
    }
    */
    let bindings = bindings
        .generate()
        .unwrap_or_else(|_| panic!("Unable to generate bindings for dav1d."));
    bindings
        .write_to_file(outfile.as_path())
        .unwrap_or_else(|_| panic!("Couldn't write bindings for dav1d"));
    println!(
        "cargo:rustc-env=CRABBYAVIF_ANDROID_NDK_MEDIA_BINDINGS_RS={}",
        outfile.display()
    );
    println!("cargo:rustc-link-lib=mediandk");
}
