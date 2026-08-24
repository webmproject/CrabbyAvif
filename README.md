# Crabby Avif 🦀

Avif parser/decoder/encoder implementation in Rust.

Feel free to file an issue for any question, suggestion or bug report.
Contributions are also welcome, see [CONTRIBUTING](CONTRIBUTING.md).

## Features

 * Supports aom as the underlying AV1 encoder.
 * Supports dav1d or android mediacodec as the underlying AV1 decoder.
 * C API compatible with [libavif](https://github.com/aomediacodec/libavif)
 * ..and more

## Build

```sh
git clone https://github.com/webmproject/CrabbyAvif.git
# If dav1d system library can be found with pkg-config, this step can be skipped.
cd CrabbyAvif/sys/dav1d-sys
./dav1d.cmd
# If libyuv system library can be found with pkg-config, this step can be skipped.
cd ../libyuv-sys
./libyuv.cmd
cd ../..
cargo build
```

For encoder support:

```sh
# If aom system library can be found with pkg-config, this step can be skipped.
cd sys/aom-sys
./aom.cmd
cd ../..
cargo build --features aom
```

## Binary

The library and package is named "crabby_avif". The name of the binary tool is
"crabbyavif".

```sh
cargo run --features aom,cli -- --help

# Example encode
cargo run --features aom,cli -- input.png -q 60 output.avif

# Example decode
cargo run --features cli -- input.avif output.png

# Example inspection
cargo run --no-default-features --features cli -- --info input.avif
```

## Tests

```sh
cargo test -- --skip test_conformance
```

### Conformance Tests

```sh
git clone https://github.com/AOMediaCodec/av1-avif.git external/av1-avif
git clone https://github.com/AOMediaCodec/libavif.git external/libavif
cd external/libavif/ext
./dav1d.cmd
cd ..
cmake -S . -B build -DAVIF_CODEC_DAV1D=LOCAL -DAVIF_LIBYUV=OFF -DAVIF_BUILD_APPS=ON
cmake --build build --parallel -t avifdec
cd ../..
cargo test -- test_conformance
```

If you already have the `av1-avif` repository checked out and the `avifdec`
binary available, you can point to those by setting the following environment
variables:

``sh
CRABBYAVIF_CONFORMANCE_TEST_DATA_DIR=<path> CRABBYAVIF_CONFORMANCE_TEST_AVIFDEC=<avifdec_binary> cargo test -- test_conformance
``

### C API Tests

```sh
# Build google test
cd external
./googletest.cmd
cd ..
# Build the library with the C API enabled and a codec
cargo build --features aom,capi --release
# Build the C/C++ Tests
cmake -S c_api_tests -B c_build
cmake --build c_build --parallel
# Run the C/C++ Tests (conformance_tests exclusive)
ctest --test-dir c_build --output-on-failure -E conformance_tests
```

### Android Tests

The decoder tests can be run on Android using [dinghy](https://crates.io/crates/cargo-dinghy).

```sh
# One time set up
cargo install cargo-dinghy
# Set path to NDK
export ANDROID_NDK_HOME=<path_to_ndk>
# Install rust toolchain for target
rustup target add aarch64-linux-android
# End of One time set up
# Make sure the device/emulator is available via adb.
cargo dinghy -d android test --no-default-features --features android_mediacodec,libyuv --target aarch64-linux-android --test decoder_tests
```

## License

See the [Apache v2.0 license](LICENSE) file.
