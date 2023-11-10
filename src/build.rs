fn main() {
    println!("cargo:rustc-link-search=/Users/vigneshv/code/dav1d/build/src");
    println!("cargo:rustc-link-lib=static=dav1d");
}