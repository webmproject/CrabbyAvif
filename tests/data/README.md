This directory contains avif test files for CrabbyAvif

## white_iden_irot.avif

- Item 1: AV1 item.
- Item 2 (primary): Identity item with irot property pointing to Item 1.

Created with:
```bash
$ MP4Box -add-derived-image :type=iden:ref=dimg,1:rotation=90:image-size=2x2 white_2x2.avif -out white_iden_irot.avif

$ MP4Box -set-primary 2 white_iden_irot.avif
```

## white_iden_chain_with_imir.avif

- Item 1: AV1 item.
- Item 2: Identity item with imir property pointing to Item 1.
- Item 3: Identity item pointing to Item 2.
- Item 4 (primary): Identity item pointing to Item 3.

Created with:
```bash
$ MP4Box -add-derived-image :type=iden:ref=dimg,1:mirror-axis=vertical:image-size=2x2 -add-derived-image :type=iden:ref=dimg,2:image-size=2x2 -add-derived-image :type=iden:ref=dimg,3:image-size=2x2 white_2x2.avif -out white_iden_chain_with_imir.avif

$ MP4Box -set-primary 4 white_iden_chain_with_imir.avif
```

## white_iden_chain_with_imir_primary.avif

- Item 1: AV1 item.
- Item 2: Identity item pointing to Item 1.
- Item 3: Identity item pointing to Item 2.
- Item 4 (primary): Identity item with imir property pointing to Item 3.

Created with:
```bash
$ MP4Box -add-derived-image :type=iden:ref=dimg,1:image-size=2x2 -add-derived-image :type=iden:ref=dimg,2:image-size=2x2 -add-derived-image :type=iden:ref=dimg,3:mirror-axis=vertical:image-size=2x2 white_2x2.avif -out white_iden_chain_with_imir_primary.avif

$ MP4Box -set-primary 4 white_iden_chain_with_imir_primary.avif
```

## white_iden_cycle.avif

- Item 1: AV1 item.
- Item 2: Identity item with imir property pointing to Item 4.
- Item 3: Identity item pointing to Item 2.
- Item 4 (primary): Identity item pointing to Item 3.

Created with:
```bash
$ MP4Box -add-derived-image :type=iden:ref=dimg,4:image-size=2x2 -add-derived-image :type=iden:ref=dimg,2:image-size=2x2 -add-derived-image :type=iden:ref=dimg,3:image-size=2x2 white_2x2.avif -out white_iden_cycle.avif

$ MP4Box -set-primary 4 white_iden_cycle.avif
```

## white_iden_self.avif

- Item 1: AV1 item.
- Item 2: Identity item pointing to Item 2..

Created with:
```bash
$ MP4Box -add-derived-image :type=iden:ref=dimg,2:image-size=2x2 -add-derived-image white_2x2.avif -out white_iden_self.avif

$ MP4Box -set-primary 2 white_iden_self.avif
```

## pyramid_pymd.avif

- Item 1, 2 and 3: AV1 items of 128x128, 256x256 and 512x512 pixels.
- A 'grpl' box with one 'pymd' EntityToGroupBox listing the three items.

The 'pymd' box carries the grouping type specific fields tile_size_x, tile_size_y and the per layer
data after the entity_id array, so the child box is 22 bytes longer than the generic
EntityToGroupBox fields alone.

Created from three gradient PNG layers with libheif built from source at commit 1a3583b with
`ENABLE_EXPERIMENTAL_FEATURES=ON`:
```bash
$ heif-enc --add-pyramid-group -A layer0.png layer1.png layer2.png -o pyramid_pymd.avif
```

Byte for byte the same file as tests/data/pyramid_pymd.avif in libavif.

## circle_auxl_two_targets.avif

- Item 1 (primary): AV1 color item.
- Item 2: AV1 alpha item, with an `auxl` item reference listing two targets: item 1 and item 3.
- Item 3: Exif item.

Section 8.11.12.1 of ISO/IEC 14496-12 represents the items linked to as an array of
to_item_IDs, so one `auxl` box can name several targets.

Taken from libavif, where it was added in
https://github.com/AOMediaCodec/libavif/pull/3331 for the same defect.

## colors-animated-8bpc-alpha-auxl-two-boxes.avif

Derived from colors-animated-8bpc-alpha-exif-xmp.avif. The alpha track 2 references the color
track 1 in one 'auxl' box and track 3 in a second 'auxl' box of its 'tref' box. The 20 byte 'tref'
box and the 44 byte 'edts' box of that track, at file offset 1586, were overwritten in place with
these 64 bytes, so the file size and every offset stay the same:

```
00 00 00 20 't' 'r' 'e' 'f'  00 00 00 0c 'a' 'u' 'x' 'l'  00 00 00 01
                             00 00 00 0c 'a' 'u' 'x' 'l'  00 00 00 03
00 00 00 20 'f' 'r' 'e' 'e'  followed by 24 zero bytes
```

## colors-animated-8bpc-alpha-auxl-two-track-ids.avif

Same as above, but with one 'auxl' box listing track 3 and then the color track 1:

```
00 00 00 18 't' 'r' 'e' 'f'  00 00 00 10 'a' 'u' 'x' 'l'  00 00 00 03  00 00 00 01
00 00 00 28 'f' 'r' 'e' 'e'  followed by 32 zero bytes
```

## colors-animated-8bpc-alpha-1x1-frame.avif

Derived from colors-animated-8bpc-alpha-exif-xmp.avif. The second sample of the 150x150 alpha track
2 (25 bytes at file offset 7194, according to the 'stco' and 'stsz' boxes of that track) was
overwritten in place with the 23 byte 1x1 AV1 payload of white_1x1.avif (at file offset 282,
according to its 'iloc' box), followed by an empty padding OBU (`7a 00`). The file size and every
offset stay the same. Decoding the second frame fails because its alpha plane does not match the
size of the first alpha frame. This is used to check that the decoded image is not left pointing to
a released alpha frame buffer.

```python
data = bytearray(open("colors-animated-8bpc-alpha-exif-xmp.avif", "rb").read())
white_1x1 = open("white_1x1.avif", "rb").read()
data[7194:7194 + 25] = white_1x1[282:282 + 23] + b"\x7a\x00"
open("colors-animated-8bpc-alpha-1x1-frame.avif", "wb").write(data)
```

## colors-animated-8bpc-alpha-prem-two-targets.avif

Derived from colors-animated-8bpc-alpha-exif-xmp.avif. The 44 byte 'edts' box of the color track 1,
at file offset 806, was overwritten in place with a 'tref' box whose 'prem' box names track 2 and
track 3:

```
00 00 00 18 't' 'r' 'e' 'f'  00 00 00 10 'p' 'r' 'e' 'm'  00 00 00 02  00 00 00 03
00 00 00 14 'f' 'r' 'e' 'e'  followed by 12 zero bytes
```

## thumbnails_grid_per_tile_alpha.avif

Derived from the coded AV1 payloads of
`color_grid_alpha_grid_gainmap_nogrid.avif` (without the `tmap` brand or item):

- Item 1 (primary): 128x200 AV1 color item.
- Item 2 (thumbnail 0, `thmb` -> 1): 2x1 `grid` color item (256x200)
  referencing 128x200 AV1 color tile items 3 and 4 (`dimg`), with per-tile
  128x200 AV1 auxiliary alpha items 5 and 6 attached directly via `auxl`
  (`5 -> 3`, `6 -> 4`) and no `grid` alpha item.
- Item 7 (thumbnail 1, `thmb` -> 1): 64x80 standalone AV1 color item without
  alpha.

## thumbnails_progressive_lsel.avif

Contains a 16x16 2-layer progressive AV1 color stream (layer 0 = 58 bytes at
quality 2, layer 1 = 302 bytes at quality 90, encoded with
`crabbyavif --progressive -q 90 --speed 10`) shared across three items:

- Item 1 (primary): 16x16 AV1 color item with `a1lx`.
- Item 2 (thumbnail 0, `thmb` -> 1): 16x16 AV1 color item with `a1lx` and no
  `lsel`.
- Item 3 (thumbnail 1, `thmb` -> 1): 16x16 AV1 color item with `a1lx` and
  `lsel` selecting layer 0.

## region/region_grid_{400,420,422,444}_{8,10,12}bpc[_alpha].avif

262x262 3x3 `grid` images used by the region decoding tests. The cells are
88x88, so the right-most column and the bottom-most row are trimmed to 86
pixels by the grid. There is one file per combination of YUV format (4:0:0,
4:2:0, 4:2:2, 4:4:4), bit depth (8, 10, 12) and alpha (`_alpha` files have a
3x3 `grid` alpha auxiliary image).

The source images were generated with ffmpeg (a test pattern, with an alpha
gradient for the `_alpha` variant, so that all the samples depend on their
position):

```bash
$ ffmpeg -f lavfi -i testsrc2=size=262x262:rate=1 -frames:v 1 -pix_fmt rgb24 \
    region_grid.png
$ ffmpeg -f lavfi -i testsrc2=size=262x262:rate=1 -frames:v 1 \
    -vf "format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='255*(X+Y)/522'" \
    -pix_fmt rgba region_grid_alpha.png
```

and encoded with avifenc (libavif 1.4.2, aom 3.15.2):

```bash
$ for y in 400 420 422 444; do for d in 8 10 12; do for a in "" _alpha; do
    avifenc -q 60 --qalpha 60 -s 10 -d ${d} -y ${y} --grid 3x3 \
      region_grid${a}.png region/region_grid_${y}_${d}bpc${a}.avif
  done; done; done
```

## region/region_clap_irot_imir.avif

120x160 4:2:0 8-bit image with alpha, a clean aperture (`clap`) derived from
the crop rectangle (12, 14, 96, 132), a rotation (`irot`) of 1 and a mirroring
(`imir`) along axis 0, used by the region decoding tests.

The source image was generated with ffmpeg (a test pattern with an alpha
gradient, so that all the samples depend on their position):

```bash
$ ffmpeg -f lavfi -i testsrc2=size=120x160:rate=1 -frames:v 1 \
    -vf "format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='255*(X+Y)/278'" \
    -pix_fmt rgba region_clap.png
```

and encoded with avifenc (libavif 1.4.2, aom 3.15.2):

```bash
$ avifenc -q 60 --qalpha 60 -s 10 -d 8 -y 420 --crop 12,14,96,132 --irot 1 \
    --imir 0 region_clap.png region/region_clap_irot_imir.avif
```

## gainmap_larger_than_base_*.avif

Images with a `tmap` gain map that is larger than the base image, used to
verify that such gain maps are ignored (b/583496039):

- `gainmap_larger_than_base_wider.avif`: 16x16 base image, 32x16 gain map.
- `gainmap_larger_than_base_taller.avif`: 16x16 base image, 16x32 gain map.
- `gainmap_larger_than_base_wider_and_taller.avif`: 16x16 base image, 64x64
  gain map.
- `gainmap_larger_than_base_grid.avif`: 128x64 base image (2x1 grid of 64x64
  cells) and a 2x1 grid of 128x128 gain map cells. The `ispe` of the gain map
  grid item was patched from 256x128 to 128x64 (the base image size), while
  its `ImageGrid` output size remains 256x128.

The first three files were generated with the CrabbyAvif encoder using a
temporary test in `tests/encoder.rs`:

```rust
for (name, gw, gh) in [("wider", 32, 16), ("taller", 16, 32),
                       ("wider_and_taller", 64, 64)] {
    let image = generate_gradient_image(16, 16, 8, PixelFormat::Yuv420,
                                        YuvRange::Full, false)?;
    let gainmap = GainMap {
        image: generate_gradient_image(gw, gh, 8, PixelFormat::Yuv420,
                                       YuvRange::Full, false)?,
        metadata: gainmap_metadata(false),
        ..Default::default()
    };
    let settings =
        encoder::Settings { speed: Some(10), ..Default::default() };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_gainmap(&image, &gainmap)?;
    std::fs::write(format!("gainmap_larger_than_base_{name}.avif"),
                   &encoder.finish()?).unwrap();
}
```

The grid file was generated similarly with:

```rust
let cell = generate_gradient_image(64, 64, 8, PixelFormat::Yuv420,
                                   YuvRange::Full, false)?;
let gainmap = GainMap {
    image: generate_gradient_image(128, 128, 8, PixelFormat::Yuv420,
                                   YuvRange::Full, false)?,
    metadata: gainmap_metadata(false),
    ..Default::default()
};
let settings = encoder::Settings { speed: Some(10), ..Default::default() };
let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
encoder.add_image_gainmap_grid(2, 1, &[&cell, &cell],
                               &[&gainmap, &gainmap])?;
std::fs::write("gainmap_larger_than_base_grid.avif",
               &encoder.finish()?).unwrap();
```

followed by overwriting the width and height of the 256x128 `ispe` box with
128 and 64.
