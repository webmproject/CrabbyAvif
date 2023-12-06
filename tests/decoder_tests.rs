use crabby_avif::*;

fn get_test_file(filename: &str) -> String {
    let project_root = env!("CARGO_MANIFEST_DIR");
    String::from(format!("{project_root}/tests/data/{filename}"))
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
    let image = res.unwrap();
    assert!(image.alpha_present);
    assert!(!image.image_sequence_track_present);
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
    let image = res.unwrap();
    assert!(!image.alpha_present);
    assert!(image.image_sequence_track_present);
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
    let image = res.unwrap();
    assert!(!image.alpha_present);
    // This will be reported as true irrespective of the preferred source.
    assert!(image.image_sequence_track_present);
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
    let image = res.unwrap();
    assert!(image.alpha_present);
    assert!(!image.image_sequence_track_present);
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
    let mut filename_with_prefix = String::from("progressive/");
    filename_with_prefix.push_str(filename);
    let mut decoder = get_decoder(&filename_with_prefix);

    decoder.settings.allow_progressive = false;
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();
    assert!(matches!(
        image.progressive_state,
        decoder::ProgressiveState::Available
    ));

    decoder.settings.allow_progressive = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();
    assert!(matches!(
        image.progressive_state,
        decoder::ProgressiveState::Active
    ));
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(decoder.image_count, layer_count);
    for _i in 0..decoder.image_count {
        let res = decoder.next_image();
        assert!(res.is_ok());
        // let _image = res.unwrap();
        // TODO: Check width and height after scaling is implemented.
        // assert_eq!(image.image.width, width);
        // assert_eq!(image.image.height, height);
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
    let image = res.unwrap();

    assert_eq!(image.icc.len(), 596);
    assert_eq!(image.icc[0], 0);
    assert_eq!(image.icc[1], 0);
    assert_eq!(image.icc[2], 2);
    assert_eq!(image.icc[3], 84);

    assert!(image.exif.is_empty());
    assert!(image.xmp.is_empty());

    decoder.settings.ignore_xmp = false;
    decoder.settings.ignore_exif = false;
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();

    assert_eq!(image.exif.len(), 1126);
    assert_eq!(image.exif[0], 73);
    assert_eq!(image.exif[1], 73);
    assert_eq!(image.exif[2], 42);
    assert_eq!(image.exif[3], 0);

    assert_eq!(image.xmp.len(), 3898);
    assert_eq!(image.xmp[0], 60);
    assert_eq!(image.xmp[1], 63);
    assert_eq!(image.xmp[2], 120);
    assert_eq!(image.xmp[3], 112);
}

// From avifgainmaptest.cc
#[test]
fn color_grid_gainmap_different_grid() {
    let mut decoder = get_decoder("color_grid_gainmap_different_grid.avif");
    decoder.settings.enable_decoding_gainmap = true;
    decoder.settings.enable_parsing_gainmap_metadata = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();
    // Color+alpha: 4x3 grid of 128x200 tiles.
    assert_eq!(image.width, 128 * 4);
    assert_eq!(image.height, 200 * 3);
    assert_eq!(image.depth, 10);
    // Gain map: 2x2 grid of 64x80 tiles.
    assert!(decoder.gainmap_present);
    assert_eq!(decoder.gainmap.image.width, 64 * 2);
    assert_eq!(decoder.gainmap.image.height, 80 * 2);
    assert_eq!(decoder.gainmap.image.depth, 8);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.0, 6);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.1, 2);
    let res = decoder.next_image();
    assert!(res.is_ok());
}

// From avifgainmaptest.cc
#[test]
fn color_grid_alpha_grid_gainmap_nogrid() {
    let mut decoder = get_decoder("color_grid_alpha_grid_gainmap_nogrid.avif");
    decoder.settings.enable_decoding_gainmap = true;
    decoder.settings.enable_parsing_gainmap_metadata = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();
    // Color+alpha: 4x3 grid of 128x200 tiles.
    assert_eq!(image.width, 128 * 4);
    assert_eq!(image.height, 200 * 3);
    assert_eq!(image.depth, 10);
    // Gain map: single image of size 64x80.
    assert!(decoder.gainmap_present);
    assert_eq!(decoder.gainmap.image.width, 64);
    assert_eq!(decoder.gainmap.image.height, 80);
    assert_eq!(decoder.gainmap.image.depth, 8);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.0, 6);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.1, 2);
    let res = decoder.next_image();
    assert!(res.is_ok());
}

// From avifgainmaptest.cc
#[test]
fn color_nogrid_alpha_nogrid_gainmap_grid() {
    let mut decoder = get_decoder("color_nogrid_alpha_nogrid_gainmap_grid.avif");
    decoder.settings.enable_decoding_gainmap = true;
    decoder.settings.enable_parsing_gainmap_metadata = true;
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();
    // Color+alpha: single image of size 128x200.
    assert_eq!(image.width, 128);
    assert_eq!(image.height, 200);
    assert_eq!(image.depth, 10);
    // Gain map: 2x2 grid of 64x80 tiles.
    assert!(decoder.gainmap_present);
    assert_eq!(decoder.gainmap.image.width, 64 * 2);
    assert_eq!(decoder.gainmap.image.height, 80 * 2);
    assert_eq!(decoder.gainmap.image.depth, 8);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.0, 6);
    assert_eq!(decoder.gainmap.metadata.alternate_hdr_headroom.1, 2);
    let res = decoder.next_image();
    assert!(res.is_ok());
}

// From avifcllitest.cc
#[test_case::test_case("clli_0_0.avif", 0, 0; "clli_0_0")]
#[test_case::test_case("clli_0_1.avif", 0, 1; "clli_0_1")]
#[test_case::test_case("clli_0_65535.avif", 0, 65535; "clli_0_65535")]
#[test_case::test_case("clli_1_0.avif", 1, 0; "clli_1_0")]
#[test_case::test_case("clli_1_1.avif", 1, 1; "clli_1_1")]
#[test_case::test_case("clli_1_65535.avif", 1, 65535; "clli_1_65535")]
#[test_case::test_case("clli_65535_0.avif", 65535, 0; "clli_65535_0")]
#[test_case::test_case("clli_65535_1.avif", 65535, 1; "clli_65535_1")]
#[test_case::test_case("clli_65535_65535.avif", 65535, 65535; "clli_65535_65535")]
fn clli(filename: &str, max_cll: u16, max_pall: u16) {
    let mut filename_with_prefix = String::from("clli/");
    filename_with_prefix.push_str(filename);
    let mut decoder = get_decoder(&filename_with_prefix);
    let res = decoder.parse();
    assert!(res.is_ok());
    let image = res.unwrap();
    if max_cll == 0 && max_pall == 0 {
        assert!(image.clli.is_none());
    } else {
        assert!(image.clli.is_some());
        let clli = image.clli.as_ref().unwrap();
        assert_eq!(clli.max_cll, max_cll);
        assert_eq!(clli.max_pall, max_pall);
    }
}
