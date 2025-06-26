// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(feature = "capi")]
use std::path::PathBuf;

#[cfg(feature = "capi")]
const LIBAVIF_COMPAT_FUNCTIONS: &[&str] = &[
    "avifAlloc",
    "avifCleanApertureBoxConvertCropRect",
    "avifCodecChoiceFromName",
    "avifCodecName",
    "avifCropRectConvertCleanApertureBox",
    "avifDecoderCreate",
    "avifDecoderDecodedRowCount",
    "avifDecoderDestroy",
    "avifDecoderIsKeyframe",
    "avifDecoderNearestKeyframe",
    "avifDecoderNextImage",
    "avifDecoderNthImage",
    "avifDecoderNthImageMaxExtent",
    "avifDecoderNthImageTiming",
    "avifDecoderParse",
    "avifDecoderRead",
    "avifDecoderReadFile",
    "avifDecoderReadMemory",
    "avifDecoderReset",
    "avifDecoderSetIO",
    "avifDecoderSetIOFile",
    "avifDecoderSetIOMemory",
    "avifDecoderSetSource",
    "avifDiagnosticsClearError",
    "avifEncoderAddImage",
    "avifEncoderAddImageGrid",
    "avifEncoderCreate",
    "avifEncoderDestroy",
    "avifEncoderFinish",
    "avifEncoderSetCodecSpecificOption",
    "avifEncoderWrite",
    "avifFree",
    "avifGainMapCreate",
    "avifGainMapDestroy",
    "avifGetPixelFormatInfo",
    "avifIOCreateFileReader",
    "avifIOCreateMemoryReader",
    "avifIODestroy",
    "avifImageAllocatePlanes",
    "avifImageCopy",
    "avifImageCreate",
    "avifImageCreateEmpty",
    "avifImageDestroy",
    "avifImageFreePlanes",
    "avifImageIsOpaque",
    "avifImagePlane",
    "avifImagePlaneHeight",
    "avifImagePlaneRowBytes",
    "avifImagePlaneWidth",
    "avifImageRGBToYUV",
    "avifImageScale",
    "avifImageSetMetadataExif",
    "avifImageSetMetadataXMP",
    "avifImageSetProfileICC",
    "avifImageSetViewRect",
    "avifImageUsesU16",
    "avifImageYUVToRGB",
    "avifPeekCompatibleFileType",
    "avifPixelFormatToString",
    "avifRGBFormatChannelCount",
    "avifRGBFormatHasAlpha",
    "avifRGBImageAllocatePixels",
    "avifRGBImageFreePixels",
    "avifRGBImagePixelSize",
    "avifRGBImageSetDefaults",
    "avifRWDataFree",
    "avifRWDataRealloc",
    "avifRWDataSet",
    "avifResultToString",
];

#[cfg(feature = "capi")]
const LIBAVIF_COMPAT_CONSTANTS: &[&str] = &[
    "AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE",
    "AVIF_FALSE",
    "AVIF_PLANE_COUNT_YUV",
    "AVIF_REPETITION_COUNT_INFINITE",
    "AVIF_REPETITION_COUNT_UNKNOWN",
    "AVIF_TRUE",
    "AVIF_DEFAULT_IMAGE_COUNT_LIMIT",
    "AVIF_DEFAULT_IMAGE_DIMENSION_LIMIT",
    "AVIF_DEFAULT_IMAGE_SIZE_LIMIT",
    "AVIF_MAX_AV1_LAYER_COUNT",
];

// In C++ mode, cbindgen does not use the struct prefix for structs. We need this so that we can
// have circular struct dependencies that use a pointer. So forward declare those structs which
// have a circular dependency.
#[cfg(feature = "capi")]
const FORWARD_DECLS: &str = r#"
namespace crabbyavif {
struct avifImage;
struct avifIO;

constexpr static uint16_t AVIF_COLOR_PRIMARIES_BT709 = 1;
constexpr static uint16_t AVIF_COLOR_PRIMARIES_IEC61966_2_4 = 1;
constexpr static uint16_t AVIF_COLOR_PRIMARIES_BT2100 = 9;
constexpr static uint16_t AVIF_COLOR_PRIMARIES_DCI_P3 = 12;
constexpr static uint16_t AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084 = 16;
}
"#;

#[cfg(feature = "capi")]
const FORWARD_DECLS_NO_NAMESPACE: &str = r#"
struct avifImage;
struct avifIO;

// In the libavif drop-in header, define them as macros to avoid unnecessary cast warnings.
#define AVIF_COLOR_PRIMARIES_BT709 AVIF_COLOR_PRIMARIES_SRGB
#define AVIF_COLOR_PRIMARIES_IEC61966_2_4 AVIF_COLOR_PRIMARIES_SRGB
#define AVIF_COLOR_PRIMARIES_BT2100 AVIF_COLOR_PRIMARIES_BT2020
#define AVIF_COLOR_PRIMARIES_DCI_P3 AVIF_COLOR_PRIMARIES_SMPTE432
#define AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084 AVIF_TRANSFER_CHARACTERISTICS_PQ
"#;

// In C++ mode, cbindgen balks on use of "Box" objects without this. This workaround of aliasing
// Box to T* comes from
// https://github.com/mozilla/cbindgen/blob/f1d5801d3b299fa2e87d176f03b605532f931cb6/tests/rust/box.toml.
#[cfg(feature = "capi")]
const AFTER_INCLUDES: &str = r#"
template <typename T>
using Box = T*;

// Used to initialize avifROData/avifRWData on the stack.
#define AVIF_DATA_EMPTY { NULL, 0 }
"#;

// cbindgen puts the trailer after the include guard, so we need to re-include the include guard
// here.
#[cfg(feature = "capi")]
const CXX_WRAPPERS: &str = r#"
#ifndef AVIF_H_CXX
#define AVIF_H_CXX

#if !defined(__cplusplus)
#error "This a C++ only header."
#endif

#include <memory>

namespace avif
{

// Struct to call the destroy functions in a unique_ptr.
struct UniquePtrDeleter
{
    void operator()(avifEncoder * encoder) const { avifEncoderDestroy(encoder); }
    void operator()(avifDecoder * decoder) const { avifDecoderDestroy(decoder); }
    void operator()(avifImage * image) const { avifImageDestroy(image); }
    void operator()(avifGainMap * gainMap) const { avifGainMapDestroy(gainMap); }
};

// Use these unique_ptr to ensure the structs are automatically destroyed.
using EncoderPtr = std::unique_ptr<avifEncoder, UniquePtrDeleter>;
using DecoderPtr = std::unique_ptr<avifDecoder, UniquePtrDeleter>;
using ImagePtr = std::unique_ptr<avifImage, UniquePtrDeleter>;
using GainMapPtr = std::unique_ptr<avifGainMap, UniquePtrDeleter>;

}

#endif // AVIF_H_CXX
"#;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    #[cfg(feature = "libgav1")]
    {
        // libgav1 needs libstdc++ on *nix/windows and libc++ on mac.
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }
    #[cfg(feature = "capi")]
    {
        let crate_path = env!("CARGO_MANIFEST_DIR");

        // Generate the C header.
        let mut config = cbindgen::Config::from_root_or_default(crate_path);
        config.after_includes = Some(FORWARD_DECLS.to_string() + AFTER_INCLUDES);
        let header_file = PathBuf::from(crate_path).join("include/avif/avif.h");
        cbindgen::Builder::new()
            .with_crate(crate_path)
            .with_config(config.clone())
            .generate()
            .unwrap()
            .write_to_file(header_file);

        // Generate the libavif compatible C header. This is the same as the C++ header above but
        // with the following modifications:
        // * No namespace.
        // * All functions are #define'd without the "crabby_" prefix.
        // * All constants are #define'd without the "CRABBY_" prefix.
        // * C++ unique_ptr wrappers (to provide compatibility for avif_cxx.h).
        config.namespace = None;
        config.after_includes = Some(FORWARD_DECLS_NO_NAMESPACE.to_string() + AFTER_INCLUDES);

        let function_redefinitions: String = LIBAVIF_COMPAT_FUNCTIONS
            .iter()
            .map(|s| format!("#define {} crabby_{}\n", s, s))
            .collect::<Vec<String>>()
            .join("");
        config
            .after_includes
            .as_mut()
            .unwrap()
            .push_str(&function_redefinitions);

        let constant_redefinitions: String = LIBAVIF_COMPAT_CONSTANTS
            .iter()
            .map(|s| format!("#define {} CRABBY_{}\n", s, s))
            .collect::<Vec<String>>()
            .join("");
        config
            .after_includes
            .as_mut()
            .unwrap()
            .push_str(&constant_redefinitions);

        config.trailer = Some(CXX_WRAPPERS.to_string());

        let header_file = PathBuf::from(crate_path).join("include/avif/avif_compat.h");
        cbindgen::Builder::new()
            .with_crate(crate_path)
            .with_config(config)
            .generate()
            .unwrap()
            .write_to_file(header_file);
    }
}
