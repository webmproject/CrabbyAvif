fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // path to dav1d .a file.
    println!("cargo:rustc-link-search=/Users/vigneshv/code/dav1d/build/src");
    // path to gav1 .a file.
    println!("cargo:rustc-link-search=/Users/vigneshv/code/libavif/ext/libgav1/build");
    println!("cargo:rustc-link-lib=static=gav1");
    println!("cargo:rustc-link-lib=static=dav1d");
}
