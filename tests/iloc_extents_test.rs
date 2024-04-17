#[path = "./mod.rs"]
mod tests;

use crabby_avif::reformat::rgb::*;
use image::io::Reader as ImageReader;
use tests::*;

#[test]
fn iloc_extents() {
    let mut decoder = get_decoder("sacre_coeur_2extents.avif");
    assert!(decoder.parse().is_ok());
    if !HAS_DECODER {
        return;
    }
    assert!(decoder.next_image().is_ok());
    let decoded = decoder.image().expect("image was none");
    let mut rgb = Image::create_from_yuv(decoded);
    rgb.format = Format::Rgb;
    assert!(rgb.allocate().is_ok());
    assert!(rgb.convert_from_yuv(decoded).is_ok());

    let source = ImageReader::open(get_test_file("sacre_coeur.png"));
    let source = source.unwrap().decode().unwrap();

    // sacre_coeur_2extents.avif was generated with
    //   avifenc --lossless --ignore-exif --ignore-xmp --ignore-icc sacre_coeur.png
    // so pixels can be compared byte by byte.
    assert_eq!(
        source.as_bytes(),
        rgb.pixels
            .as_ref()
            .unwrap()
            .slice(0, source.as_bytes().len() as u32)
            .unwrap()
    );
}

#[test]
fn nth_image_max_extent() {
    let mut decoder = get_decoder("sacre_coeur_2extents.avif");
    assert!(decoder.parse().is_ok());

    let max_extent = decoder.nth_image_max_extent(0).unwrap();
    assert_eq!(max_extent.offset, 290);
    assert_eq!(max_extent.size, 1000 + 1 + 5778); // '\0' in the middle.
}
