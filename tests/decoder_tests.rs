use rust_libavif::*;

const TEST_DATA_PATH: &str = "/Users/vigneshv/code/rust_pg/rust-libavif/tests/data";

fn get_test_file(filename: &str) -> String {
    String::from(format!("{TEST_DATA_PATH}/{filename}"))
}

fn get_decoder(filename: &str) -> decoder::Decoder {
    let abs_filename = get_test_file(filename);
    let mut decoder = decoder::Decoder::default();
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
    assert!(matches!(
        decoder.settings.strictness,
        decoder::Strictness::All
    ));
    let res = decoder.parse();
    assert_avif_error!(res, BmffParseFailed);
    // Allow this kind of file specifically.
    decoder.settings.strictness =
        decoder::Strictness::SpecificExclude(vec![decoder::StrictnessFlag::AlphaIspeRequired]);
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
    decoder.settings.source = decoder::Source::PrimaryItem;
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
    assert!(info.alpha_present);
    assert!(!info.image_sequence_track_present);
    let res = decoder.next_image();
    assert!(res.is_ok());
    let image = res.unwrap();
    let alpha_plane = image.plane(3);
    assert!(alpha_plane.is_some());
    assert!(alpha_plane.unwrap().row_bytes > 0);
}

// From avifprogressivetest.cc
#[test_case::test_case("progressive_dimension_change.avif", 2, 256, 256; "progressive_dimension_change")]
#[test_case::test_case("progressive_layered_grid.avif", 2, 512, 256; "progressive_layered_grid")]
#[test_case::test_case("progressive_quality_change.avif", 2, 256, 256; "progressive_quality_change")]
#[test_case::test_case("progressive_same_layers.avif", 4, 256, 256; "progressive_same_layers")]
fn progressive(filename: &str, layer_count: u32, width: u32, height: u32) {
    let mut decoder = get_decoder(filename);

    decoder.settings.allow_progressive = false;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    assert!(matches!(
        info.progressive_state,
        decoder::ProgressiveState::Available
    ));

    decoder.settings.allow_progressive = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    assert!(matches!(
        info.progressive_state,
        decoder::ProgressiveState::Active
    ));
    assert_eq!(info.width, width);
    assert_eq!(info.height, height);
    assert_eq!(decoder.image_count, layer_count);
    for _ in 0..decoder.image_count {
        let res = decoder.next_image();
        assert!(res.is_ok());
        // let _image = res.unwrap();
        // TODO: Check width and height after scaling is implemented.
        // assert_eq!(image.info.width, width);
        // assert_eq!(image.info.height, height);
    }
}

// From avifmetadatatest.cc
#[test]
fn decoder_parse_icc_exif_xmp() {
    // Test case from https://github.com/AOMediaCodec/libavif/issues/1086.
    let mut decoder = get_decoder("paris_icc_exif_xmp.avif");

    decoder.settings.ignore_xmp = true;
    decoder.settings.ignore_exif = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();

    assert_eq!(info.icc.len(), 596);
    assert_eq!(info.icc[0], 0);
    assert_eq!(info.icc[1], 0);
    assert_eq!(info.icc[2], 2);
    assert_eq!(info.icc[3], 84);

    assert!(info.exif.is_empty());
    assert!(info.xmp.is_empty());

    decoder.settings.ignore_xmp = false;
    decoder.settings.ignore_exif = false;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();

    assert_eq!(info.exif.len(), 1126);
    assert_eq!(info.exif[0], 73);
    assert_eq!(info.exif[1], 73);
    assert_eq!(info.exif[2], 42);
    assert_eq!(info.exif[3], 0);

    assert_eq!(info.xmp.len(), 3898);
    assert_eq!(info.xmp[0], 60);
    assert_eq!(info.xmp[1], 63);
    assert_eq!(info.xmp[2], 120);
    assert_eq!(info.xmp[3], 112);
}

// From GainMaptest.cc
#[test]
fn color_grid_gainmap_different_grid() {
    let mut decoder = get_decoder("color_grid_gainmap_different_grid.avif");
    decoder.settings.enable_decoding_gainmap = true;
    decoder.settings.enable_parsing_gainmap_metadata = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    // Color+alpha: 4x3 grid of 128x200 tiles.
    assert_eq!(info.width, 128 * 4);
    assert_eq!(info.height, 200 * 3);
    assert_eq!(info.depth, 10);
    // Gain map: 2x2 grid of 64x80 tiles.
    assert!(decoder.gainmap_present);
    assert_eq!(decoder.gainmap.image.info.width, 64 * 2);
    assert_eq!(decoder.gainmap.image.info.height, 80 * 2);
    assert_eq!(decoder.gainmap.image.info.depth, 8);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.0, 6);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.1, 2);
    let res = decoder.next_image();
    assert!(res.is_ok());
}

// From GainMaptest.cc
#[test]
fn color_grid_alpha_grid_gainmap_nogrid() {
    let mut decoder = get_decoder("color_grid_alpha_grid_gainmap_nogrid.avif");
    decoder.settings.enable_decoding_gainmap = true;
    decoder.settings.enable_parsing_gainmap_metadata = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    // Color+alpha: 4x3 grid of 128x200 tiles.
    assert_eq!(info.width, 128 * 4);
    assert_eq!(info.height, 200 * 3);
    assert_eq!(info.depth, 10);
    // Gain map: single image of size 64x80.
    assert!(decoder.gainmap_present);
    assert_eq!(decoder.gainmap.image.info.width, 64);
    assert_eq!(decoder.gainmap.image.info.height, 80);
    assert_eq!(decoder.gainmap.image.info.depth, 8);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.0, 6);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.1, 2);
    let res = decoder.next_image();
    assert!(res.is_ok());
}

// From GainMaptest.cc
#[test]
fn color_nogrid_alpha_nogrid_gainmap_grid() {
    let mut decoder = get_decoder("color_nogrid_alpha_nogrid_gainmap_grid.avif");
    decoder.settings.enable_decoding_gainmap = true;
    decoder.settings.enable_parsing_gainmap_metadata = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let info = res.unwrap();
    // Color+alpha: single image of size 128x200.
    assert_eq!(info.width, 128);
    assert_eq!(info.height, 200);
    assert_eq!(info.depth, 10);
    // Gain map: 2x2 grid of 64x80 tiles.
    assert!(decoder.gainmap_present);
    assert_eq!(decoder.gainmap.image.info.width, 64 * 2);
    assert_eq!(decoder.gainmap.image.info.height, 80 * 2);
    assert_eq!(decoder.gainmap.image.info.depth, 8);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.0, 6);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.1, 2);
    let res = decoder.next_image();
    assert!(res.is_ok());
}
