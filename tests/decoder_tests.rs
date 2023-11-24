use rust_libavif::*;

const TEST_DATA_PATH: &str = "/Users/vigneshv/code/rust_pg/rust-libavif/tests/data";

fn get_test_file(filename: &str) -> String {
    String::from(format!("{TEST_DATA_PATH}/{filename}"))
}

fn get_decoder(filename: &str) -> decoder::AvifDecoder {
    let abs_filename = get_test_file(filename);
    let mut decoder = decoder::AvifDecoder::default();
    let _ = decoder
        .set_io_file(&abs_filename)
        .expect("Failed to set IO");
    decoder
}

macro_rules! assert_avif_error {
    ($res:ident, $err:ident) => {
        assert!($res.is_err());
        assert!(matches!($res.err().unwrap(), AvifError::$err));
    };
}

// From avifalphanoispetest.cc
#[test]
fn alpha_no_ispe() {
    // See https://github.com/AOMediaCodec/libavif/pull/745.
    let mut decoder = get_decoder("alpha_noispe.avif");
    // By default, non-strict files are refused.
    assert!(matches!(decoder.settings.strictness, AvifStrictness::All));
    let res = decoder.parse();
    assert_avif_error!(res, BmffParseFailed);
    // Allow this kind of file specifically.
    decoder.settings.strictness =
        AvifStrictness::SpecificExclude(vec![AvifStrictnessFlag::AlphaIspeRequired]);
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    assert!(info.alpha_present);
    assert!(!info.image_sequence_track_present);
    let res = decoder.next_image();
    assert!(res.is_ok());
    let image = res.unwrap();
    let alpha_plane = image.plane(3);
    assert!(alpha_plane.is_some());
    assert!(alpha_plane.unwrap().row_bytes > 0);
}

// From avifanimationtest.cc
#[test]
fn animated_image() {
    let mut decoder = get_decoder("colors-animated-8bpc.avif");
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    assert!(!info.alpha_present);
    assert!(info.image_sequence_track_present);
    assert_eq!(decoder.image_count, 5);
    assert_eq!(decoder.repetition_count, 0);
    for _ in 0..5 {
        assert!(decoder.next_image().is_ok());
    }
}

// From avifanimationtest.cc
#[test]
fn animated_image_with_source_set_to_primary_item() {
    let mut decoder = get_decoder("colors-animated-8bpc.avif");
    decoder.settings.source = decoder::AvifDecoderSource::PrimaryItem;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    assert!(!info.alpha_present);
    // This will be reported as true irrespective of the preferred source.
    assert!(info.image_sequence_track_present);
    // imageCount is expected to be 1 because we are using primary item as the
    // preferred source.
    assert_eq!(decoder.image_count, 1);
    assert_eq!(decoder.repetition_count, 0);
    // Get the first (and only) image.
    assert!(decoder.next_image().is_ok());
    // Subsequent calls should not return anything since there is only one
    // image in the preferred source.
    assert!(decoder.next_image().is_err());
}

// From avifdecodetest.cc
#[test]
fn color_grid_alpha_no_grid() {
    // Test case from https://github.com/AOMediaCodec/libavif/issues/1203.
    let mut decoder = get_decoder("color_grid_alpha_nogrid.avif");
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    // TODO: this assertion has to be reversed once such files are supported.
    assert!(!info.alpha_present);
    assert!(!info.image_sequence_track_present);
    let res = decoder.next_image();
    assert!(res.is_ok());
    // TODO: these assertions has to be reversed once such files are supported.
    let image = res.unwrap();
    let alpha_plane = image.plane(3);
    assert!(!alpha_plane.is_some());
    //assert!(alpha_plane.unwrap().row_bytes > 0);
}
