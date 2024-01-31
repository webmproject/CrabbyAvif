fn main() {
    #[cfg(feature = "libgav1")]
    {
        // libgav1 needs libstdc++ on *nix and libc++ on mac. TODO: what about windows?
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }
}
