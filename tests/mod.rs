use crabby_avif::*;

#[cfg(test)]
pub fn get_test_file(filename: &str) -> String {
    let base_path = if cfg!(google3) {
        format!(
            "{}/google3/third_party/crabbyavif/",
            std::env::var("TEST_SRCDIR").expect("TEST_SRCDIR is not defined")
        )
    } else {
        "".to_string()
    };
    String::from(format!("{base_path}tests/data/{filename}"))
}

#[cfg(test)]
pub fn get_decoder(filename: &str) -> decoder::Decoder {
    let abs_filename = get_test_file(filename);
    let mut decoder = decoder::Decoder::default();
    let _ = decoder
        .set_io_file(&abs_filename)
        .expect("Failed to set IO");
    decoder
}

#[cfg(test)]
#[allow(dead_code)]
pub const HAS_DECODER: bool = if cfg!(any(
    feature = "dav1d",
    feature = "libgav1",
    feature = "android_mediacodec"
)) {
    true
} else {
    false
};
