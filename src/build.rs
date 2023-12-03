fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // TODO: replace these paths with generic instructions on how to build the
    // dependencies.

    #[cfg(feature = "dav1d")]
    {
        println!("cargo:rustc-link-search=/Users/vigneshv/code/dav1d/build/src");
        println!("cargo:rustc-link-lib=static=dav1d");
    }

    #[cfg(feature = "libgav1")]
    {
        println!("cargo:rustc-link-search=/Users/vigneshv/code/libavif/ext/libgav1/build");
        println!("cargo:rustc-link-lib=static=gav1");
        println!("cargo:rustc-link-arg=-lc++");
    }
}
