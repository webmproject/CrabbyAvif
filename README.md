# Crabby Avif 🦀

Avif parser/decoder implementation in Rust.

## Features
 * Supports dav1d, libgav1 or android mediacodec as the underlying AV1 decoder.
 * C API compatible with [libavif](https://github.com/aomediacodec/libavif)
 * ..and more

## Build

```sh
git clone https://github.com/webmproject/CrabbyAvif.git
cd CrabbyAvif/sys/dav1d-sys
./dav1d.cmd
cd ../libyuv-sys
./libyuv.cmd
cd ../..
cargo build
```

### Tests

```sh
cargo test -- --skip test_conformance
```

#### Conformance tests

```sh
git clone https://github.com/AOMediaCodec/av1-avif.git third_party/av1-avif
git clone https://github.com/AOMediaCodec/libavif.git third_party/libavif
cd third_party/libavif/ext
./aom.cmd
cd ../../..
cmake -S third_party/libavif -B third_party/libavif/build -DAVIF_CODEC_AOM=LOCAL -DAVIF_BUILD_APPS=ON
cmake --build third_party/libavif/build --parallel
cargo test -- test_conformance
```
