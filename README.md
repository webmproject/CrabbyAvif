# Crabby Avif 🦀

Avif parser/decoder implementation in Rust.

## Features
 * Supports dav1d, libgav1 or android mediacodec as the underlying AV1 decoder.
 * C API compatible with [libavif](https://github.com/aomediacodec/libavif)
 * ..and more

## Build

```sh
git clone https://github.com/webmproject/CrabbyAvif.git
# If dav1d system library can be found with pkg-config, this step can be
skipped.
cd CrabbyAvif/sys/dav1d-sys
./dav1d.cmd
# If libyuv system library can be found with pkg-config, this step can be
skipped.
cd ../libyuv-sys
./libyuv.cmd
cd ../..
cargo build
```

## Tests

```sh
cargo test -- --skip test_conformance
```

### Conformance Tests

```sh
git clone https://github.com/AOMediaCodec/av1-avif.git third_party/av1-avif
git clone https://github.com/AOMediaCodec/libavif.git third_party/libavif
cd third_party/libavif/ext
./dav1d.cmd
cd ../../..
cmake -S third_party/libavif -B third_party/libavif/build -DAVIF_CODEC_DAV1D=LOCAL -DAVIF_BUILD_APPS=ON
cmake --build third_party/libavif/build --parallel -t avifdec
cargo test -- test_conformance
```

### C API Tests

```sh
# Build google test
cd third_party
./googletest.cmd
cd ..
# Build the library with C API enabled
cargo build --features capi --release
# Build and run the C/C++ Tests
mkdir c_build
cd c_build
cmake ../c_api_tests/
make
make test
```
