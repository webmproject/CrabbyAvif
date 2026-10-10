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

## chroma_vertical_10bit.avif

A synthetic 16×16, 10-bit YUV420 image with vertical chroma positioning, limited range, and BT.2020/PQ signaling. Both chroma planes vary horizontally and vertically. Encoded losslessly with FFmpeg 8.1.2 and libaom 3.15.0:

```bash
$ python3 - <<'PYTHON'
import struct
from pathlib import Path

y = [504] * 256
u = [384 + 8 * x + 4 * y + x * y for y in range(8) for x in range(8)]
v = [640 - 4 * x - 8 * y + x * y for y in range(8) for x in range(8)]
Path("chroma_vertical_10bit.yuv").write_bytes(struct.pack("<384H", *y, *u, *v))
PYTHON

$ ffmpeg -f rawvideo -pixel_format yuv420p10le -video_size 16x16 \
  -color_range tv -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc \
  -i chroma_vertical_10bit.yuv -frames:v 1 -c:v libaom-av1 -crf 0 -b:v 0 \
  -still-picture 1 -cpu-used 6 -aom-params chroma-sample-position=1 chroma_vertical_10bit.avif
```
