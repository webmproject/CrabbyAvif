// Copyright 2025 Google LLC
// SPDX-License-Identifier: BSD-2-Clause

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "avif/avif.h"
#include "gtest/gtest.h"
#include "testutil.h"

namespace avif {
namespace {

// Used to pass the data folder path to the GoogleTest suites.
const char* data_path = nullptr;

std::string GetFilename(const char* file_name) {
  return std::string(data_path) + file_name;
}

DecoderPtr CreateDecoder(const char* file_name) {
  DecoderPtr decoder(avifDecoderCreate());
  if (decoder == nullptr ||
      avifDecoderSetIOFile(decoder.get(), GetFilename(file_name).c_str()) !=
          AVIF_RESULT_OK) {
    return nullptr;
  }
  return decoder;
}

// Returns the AV1 decoders that the decoding tests run with: dav1d and the
// native AV1 decoder (only available when crabbyavif is built with the
// native_av1 feature). Falls back to the default decoder if neither is
// available (for example on Android, which uses MediaCodec).
std::vector<avifCodecChoice> GetCodecChoices() {
  std::vector<avifCodecChoice> codec_choices;
  for (const char* name : {"dav1d", "native"}) {
    const avifCodecChoice codec_choice = avifCodecChoiceFromName(name);
    if (codec_choice != AVIF_CODEC_CHOICE_AUTO) {
      codec_choices.push_back(codec_choice);
    }
  }
  if (codec_choices.empty()) {
    codec_choices.push_back(AVIF_CODEC_CHOICE_AUTO);
  }
  return codec_choices;
}

std::string GetCodecName(avifCodecChoice codec_choice) {
  switch (codec_choice) {
    case AVIF_CODEC_CHOICE_DAV1D:
      return "dav1d";
    case AVIF_CODEC_CHOICE_NATIVE:
      return "native";
    default:
      return "auto";
  }
}

// Returns a new decoder that uses |codec_choice|.
avifDecoder* NewDecoder(avifCodecChoice codec_choice) {
  avifDecoder* decoder = avifDecoderCreate();
  if (decoder != nullptr) {
    decoder->codecChoice = codec_choice;
  }
  return decoder;
}

DecoderPtr CreateDecoder(const char* file_name, avifCodecChoice codec_choice) {
  DecoderPtr decoder = CreateDecoder(file_name);
  if (decoder != nullptr) {
    decoder->codecChoice = codec_choice;
  }
  return decoder;
}

// Fixture of the tests that decode images, parameterized by the AV1 decoder.
class DecoderTest : public testing::TestWithParam<avifCodecChoice> {
 protected:
  avifCodecChoice codec_choice() const { return GetParam(); }
  avifDecoder* NewDecoder() const { return avif::NewDecoder(codec_choice()); }
  DecoderPtr CreateDecoder(const char* file_name) const {
    return avif::CreateDecoder(file_name, codec_choice());
  }
};

TEST_P(DecoderTest, AlphaNoIspe) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  // See https://github.com/AOMediaCodec/libavif/pull/745.
  auto decoder = CreateDecoder("alpha_noispe.avif");
  ASSERT_NE(decoder, nullptr);
  // By default, loose files are refused. Cast to avoid C4389 Windows warning.
  EXPECT_EQ(decoder->strictFlags,
            static_cast<avifStrictFlags>(AVIF_STRICT_ENABLED));
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_BMFF_PARSE_FAILED);
  // Allow this kind of file specifically.
  decoder->strictFlags =
      static_cast<avifStrictFlags>(AVIF_STRICT_ENABLED) &
      ~static_cast<avifStrictFlags>(AVIF_STRICT_ALPHA_ISPE_REQUIRED);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_NE(decoder->image->alphaPlane, nullptr);
  EXPECT_GT(decoder->image->alphaRowBytes, 0u);
}

TEST_P(DecoderTest, AlphaPremultiplied) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder("alpha_premultiplied.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  ASSERT_NE(decoder->image, nullptr);
  EXPECT_EQ(decoder->image->alphaPremultiplied, AVIF_TRUE);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_NE(decoder->image->alphaPlane, nullptr);
  EXPECT_GT(decoder->image->alphaRowBytes, 0u);
}

TEST_P(DecoderTest, AlphaPremultipliedIgnoreAlpha) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder("alpha_premultiplied.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode &= ~AVIF_IMAGE_CONTENT_ALPHA;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  ASSERT_NE(decoder->image, nullptr);
  EXPECT_EQ(decoder->image->alphaPremultiplied, AVIF_TRUE);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->image->alphaPlane, nullptr);
  EXPECT_EQ(decoder->image->alphaRowBytes, 0u);
}

TEST_P(DecoderTest, AnimatedImage) {
  if (codec_choice() == AVIF_CODEC_CHOICE_NATIVE) {
    GTEST_SKIP() << "The native AV1 decoder does not support inter frames.";
  }
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder("colors-animated-8bpc.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_FALSE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageCount, 5);
  EXPECT_EQ(decoder->repetitionCount, 0);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  }
}

TEST_P(DecoderTest, AnimatedImageWithSourceSetToPrimaryItem) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder("colors-animated-8bpc.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(
      avifDecoderSetSource(decoder.get(), AVIF_DECODER_SOURCE_PRIMARY_ITEM),
      AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_FALSE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_TRUE);
  // imageCount is expected to be 1 because we are using primary item as the
  // preferred source.
  EXPECT_EQ(decoder->imageCount, 1);
  EXPECT_EQ(decoder->repetitionCount, 0);
  // Get the first (and only) image.
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  // Subsequent calls should not return AVIF_RESULT_OK since there is only one
  // image in the preferred source.
  EXPECT_NE(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
}

TEST(DecoderTest, AnimatedImageWithAlphaAndMetadata) {
  auto decoder = CreateDecoder("colors-animated-8bpc-alpha-exif-xmp.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageCount, 5);
  EXPECT_EQ(decoder->repetitionCount, AVIF_REPETITION_COUNT_INFINITE);
  EXPECT_EQ(decoder->image->exif.size, 1126);
  EXPECT_EQ(decoder->image->xmp.size, 3898);
}

TEST(DecoderTest, AnimatedImageWithAlphaAndMetadataIgnoreAlpha) {
  auto decoder = CreateDecoder("colors-animated-8bpc-alpha-exif-xmp.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode &= ~AVIF_IMAGE_CONTENT_ALPHA;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageCount, 5);
  EXPECT_EQ(decoder->repetitionCount, AVIF_REPETITION_COUNT_INFINITE);
  EXPECT_EQ(decoder->image->exif.size, 1126);
  EXPECT_EQ(decoder->image->xmp.size, 3898);
}

TEST_P(DecoderTest, OneShotDecodeFile) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
#ifdef __ANDROID__
  // Android MediaCodec does not support re-use of same decoder instance for
  // multiple images.
  GTEST_SKIP() << "Unsupported test on Android.";
#endif
  const char* file_name = "sofa_grid1x5_420.avif";
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  avifImage image;
  ASSERT_EQ(avifDecoderReadFile(decoder.get(), &image,
                                GetFilename(file_name).c_str()),
            AVIF_RESULT_OK);
  EXPECT_EQ(image.width, 1024);
  EXPECT_EQ(image.height, 770);
  EXPECT_EQ(image.depth, 8);

  // Call avifDecoderReadFile with a different file but with the same decoder
  // instance.
  file_name = "white_1x1.avif";
  ASSERT_EQ(avifDecoderReadFile(decoder.get(), &image,
                                GetFilename(file_name).c_str()),
            AVIF_RESULT_OK);
  EXPECT_EQ(image.width, 1);
  EXPECT_EQ(image.height, 1);
  EXPECT_EQ(image.depth, 8);
}

TEST_P(DecoderTest, OneShotDecodeMemory) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const char* file_name = "sofa_grid1x5_420.avif";
  auto file_data = testutil::read_file(GetFilename(file_name).c_str());
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  avifImage image;
  ASSERT_EQ(avifDecoderReadMemory(decoder.get(), &image, file_data.data(),
                                  file_data.size()),
            AVIF_RESULT_OK);
  EXPECT_EQ(image.width, 1024);
  EXPECT_EQ(image.height, 770);
  EXPECT_EQ(image.depth, 8);
}

avifResult io_read(struct avifIO* io, uint32_t flags, uint64_t offset,
                   size_t size, avifROData* out) {
  avifROData* src = reinterpret_cast<avifROData*>(io->data);
  if (flags != 0 || offset > src->size) {
    return AVIF_RESULT_IO_ERROR;
  }
  uint64_t available_size = src->size - offset;
  if (size > available_size) {
    size = static_cast<size_t>(available_size);
  }
  out->data = src->data + offset;
  out->size = size;
  return AVIF_RESULT_OK;
}

TEST_P(DecoderTest, OneShotDecodeCustomIO) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const char* file_name = "sofa_grid1x5_420.avif";
  auto data = testutil::read_file(GetFilename(file_name).c_str());
  avifROData ro_data = {.data = data.data(), .size = data.size()};
  avifIO io = {.destroy = nullptr,
               .read = io_read,
               .sizeHint = data.size(),
               .persistent = false,
               .data = &ro_data};
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  avifDecoderSetIO(decoder.get(), &io);
  avifImage image;
  ASSERT_EQ(avifDecoderRead(decoder.get(), &image), AVIF_RESULT_OK);
  EXPECT_EQ(image.width, 1024);
  EXPECT_EQ(image.height, 770);
  EXPECT_EQ(image.depth, 8);
}

class DerivedIO : public avifIO {
 public:
  static avifIO* Create(const std::vector<uint8_t>& io_data) {
    // Manage this memory manually to ensure that crabbyavif calls the destroy
    // callback correctly.
    DerivedIO* io = new DerivedIO(io_data);
    io->destroy = DerivedIO::Destroy;
    io->read = DerivedIO::Read;
    io->sizeHint = io_data.size();
    io->persistent = AVIF_FALSE;
    return io;
  }

  static void Destroy(avifIO* io) { delete reinterpret_cast<DerivedIO*>(io); }

  static avifResult Read(avifIO* io, uint32_t flags, uint64_t offset,
                         size_t size, avifROData* out) {
    DerivedIO* derived_io = reinterpret_cast<DerivedIO*>(io);
    if (flags != 0 || offset > derived_io->data_.size()) {
      return AVIF_RESULT_IO_ERROR;
    }
    uint64_t available_size = derived_io->data_.size() - offset;
    if (size > available_size) {
      size = static_cast<size_t>(available_size);
    }
    out->data = derived_io->data_.data() + offset;
    out->size = size;
    return AVIF_RESULT_OK;
  }

 private:
  DerivedIO(const std::vector<uint8_t>& io_data) : data_(io_data) {}

  const std::vector<uint8_t>& data_;
};

TEST_P(DecoderTest, DerivedIO) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const char* file_name = "sofa_grid1x5_420.avif";
  auto data = testutil::read_file(GetFilename(file_name).c_str());
  avifIO* io = DerivedIO::Create(data);
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  avifDecoderSetIO(decoder.get(), io);
  avifImage image;
  ASSERT_EQ(avifDecoderRead(decoder.get(), &image), AVIF_RESULT_OK);
  EXPECT_EQ(image.width, 1024);
  EXPECT_EQ(image.height, 770);
  EXPECT_EQ(image.depth, 8);
}

TEST_P(DecoderTest, NthImage) {
  if (codec_choice() == AVIF_CODEC_CHOICE_NATIVE) {
    GTEST_SKIP() << "The native AV1 decoder does not support inter frames.";
  }
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder("colors-animated-8bpc.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->imageCount, 5);
  EXPECT_EQ(avifDecoderNthImage(decoder.get(), 3), AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderDecodedRowCount(decoder.get()), 150);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderNthImage(decoder.get(), 1), AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderNthImage(decoder.get(), 4), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNthImage(decoder.get(), 50), AVIF_RESULT_OK);
  for (int i = 0; i < 5; ++i) {
  }
}

TEST(DecoderTest, Clli) {
  struct Params {
    const char* file_name;
    uint32_t maxCLL;
    uint32_t maxPALL;
  };
  Params params[9] = {
      {"clli/clli_0_0.avif", 0, 0},
      {"clli/clli_0_1.avif", 0, 1},
      {"clli/clli_0_65535.avif", 0, 65535},
      {"clli/clli_1_0.avif", 1, 0},
      {"clli/clli_1_1.avif", 1, 1},
      {"clli/clli_1_65535.avif", 1, 65535},
      {"clli/clli_65535_0.avif", 65535, 0},
      {"clli/clli_65535_1.avif", 65535, 1},
      {"clli/clli_65535_65535.avif", 65535, 65535},
  };
  for (const auto& param : params) {
    DecoderPtr decoder(avifDecoderCreate());
    ASSERT_NE(decoder, nullptr);
    decoder->allowProgressive = true;
    ASSERT_EQ(avifDecoderSetIOFile(decoder.get(),
                                   GetFilename(param.file_name).c_str()),
              AVIF_RESULT_OK);
    ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
    EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
    avifImage* decoded = decoder->image;
    ASSERT_NE(decoded, nullptr);
    ASSERT_EQ(decoded->clli.maxCLL, param.maxCLL);
    ASSERT_EQ(decoded->clli.maxPALL, param.maxPALL);
  }
}

TEST_P(DecoderTest, ColorGridAlphaNoGrid) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  // Test case from https://github.com/AOMediaCodec/libavif/issues/1203.
  auto decoder = CreateDecoder("color_grid_alpha_nogrid.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_FALSE);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_NE(decoder->image->alphaPlane, nullptr);
  EXPECT_GT(decoder->image->alphaRowBytes, 0u);
}

TEST_P(DecoderTest, ColorGridAlphaNoGridIgnoreAlpha) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  // Test case from https://github.com/AOMediaCodec/libavif/issues/1203.
  auto decoder = CreateDecoder("color_grid_alpha_nogrid.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode &= ~AVIF_IMAGE_CONTENT_ALPHA;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_FALSE);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->image->alphaPlane, nullptr);
  EXPECT_EQ(decoder->image->alphaRowBytes, 0u);
}

TEST_P(DecoderTest, GainMapGrid) {
  auto decoder = CreateDecoder("color_grid_gainmap_different_grid.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode |= AVIF_IMAGE_CONTENT_GAIN_MAP;

  // Just parse the image first.
  auto result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  avifImage* decoded = decoder->image;
  ASSERT_NE(decoded, nullptr);

  // Verify that the gain map is present and matches the input.
  EXPECT_NE(decoded->gainMap, nullptr);
  // Color+alpha: 4x3 grid of 128x200 tiles.
  EXPECT_EQ(decoded->width, 128u * 4u);
  EXPECT_EQ(decoded->height, 200u * 3u);
  EXPECT_EQ(decoded->depth, 10u);
  ASSERT_NE(decoded->gainMap->image, nullptr);
  // Gain map: 2x2 grid of 64x80 tiles.
  EXPECT_EQ(decoded->gainMap->image->width, 64u * 2u);
  EXPECT_EQ(decoded->gainMap->image->height, 80u * 2u);
  EXPECT_EQ(decoded->gainMap->image->depth, 8u);
  EXPECT_EQ(decoded->gainMap->baseHdrHeadroom.n, 6u);
  EXPECT_EQ(decoded->gainMap->baseHdrHeadroom.d, 2u);

  // Decode the image.
  result = avifDecoderNextImage(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
}

TEST_P(DecoderTest, GainMapOriented) {
  auto decoder = CreateDecoder("gainmap_oriented.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode |= AVIF_IMAGE_CONTENT_GAIN_MAP;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);

  // Verify that the transformative properties were kept.
  EXPECT_EQ(decoder->image->transformFlags,
            AVIF_TRANSFORM_IROT | AVIF_TRANSFORM_IMIR);
  EXPECT_EQ(decoder->image->irot.angle, 1);
  EXPECT_EQ(decoder->image->imir.axis, 0);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->image->gainMap->image->transformFlags,
            AVIF_TRANSFORM_NONE);

  ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_NE(decoder->image->alphaPlane, nullptr);
  EXPECT_GT(decoder->image->alphaRowBytes, 0u);
}

TEST_P(DecoderTest, GainMapOrientedIgnoreAlpha) {
  auto decoder = CreateDecoder("gainmap_oriented.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode = AVIF_IMAGE_CONTENT_COLOR_AND_GAIN_MAP;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);

  // Verify that the transformative properties were kept.
  EXPECT_EQ(decoder->image->transformFlags,
            AVIF_TRANSFORM_IROT | AVIF_TRANSFORM_IMIR);
  EXPECT_EQ(decoder->image->irot.angle, 1);
  EXPECT_EQ(decoder->image->imir.axis, 0);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->image->gainMap->image->transformFlags,
            AVIF_TRANSFORM_NONE);

  ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->image->alphaPlane, nullptr);
  EXPECT_EQ(decoder->image->alphaRowBytes, 0u);
}

TEST_P(DecoderTest, GainmapOnlyDecodedRowCount) {
  // Both image and gainmap have the same height (34).
  auto decoder = CreateDecoder("gainmap_oriented.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode = AVIF_IMAGE_CONTENT_GAIN_MAP;
  auto result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK);
  result = avifDecoderNextImage(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderDecodedRowCount(decoder.get()), 34);

  // Image height is 600 and gainmap height is 160.
  decoder = CreateDecoder("color_grid_gainmap_different_grid.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode = AVIF_IMAGE_CONTENT_GAIN_MAP;
  result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK);
  result = avifDecoderNextImage(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK);
  // Image height will be returned per the API contract even though the gainmap
  // height is different.
  EXPECT_EQ(avifDecoderDecodedRowCount(decoder.get()), 600);
}

TEST(DecoderTest, IgnoreGainMapButReadMetadata) {
  auto decoder = CreateDecoder("seine_sdr_gainmap_srgb.avif");
  ASSERT_NE(decoder, nullptr);
  auto result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
  avifImage* decoded = decoder->image;
  ASSERT_NE(decoded, nullptr);

  // Verify that the gain map was detected...
  EXPECT_NE(decoded->gainMap, nullptr);
  // ... but not decoded because enableDecodingGainMap is false by default.
  EXPECT_EQ(decoded->gainMap->image, nullptr);
  // Check that the gain map metadata WAS populated.
  EXPECT_EQ(decoded->gainMap->alternateHdrHeadroom.n, 13);
  EXPECT_EQ(decoded->gainMap->alternateHdrHeadroom.d, 10);
}

TEST_P(DecoderTest, IgnoreColorAndAlpha) {
  auto decoder = CreateDecoder("seine_sdr_gainmap_srgb.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode = AVIF_IMAGE_CONTENT_GAIN_MAP;
  auto result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
  result = avifDecoderNextImage(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
  avifImage* decoded = decoder->image;
  ASSERT_NE(decoded, nullptr);

  // Main image metadata is available.
  EXPECT_EQ(decoded->width, 400u);
  EXPECT_EQ(decoded->height, 300u);
  // But pixels are not.
  EXPECT_EQ(decoded->yuvRowBytes[0], 0u);
  EXPECT_EQ(decoded->yuvRowBytes[1], 0u);
  EXPECT_EQ(decoded->yuvRowBytes[2], 0u);
  EXPECT_EQ(decoded->alphaRowBytes, 0u);
  // The gain map was decoded.
  EXPECT_NE(decoded->gainMap, nullptr);
  ASSERT_NE(decoded->gainMap->image, nullptr);
  // Including pixels.
  EXPECT_GT(decoded->gainMap->image->yuvRowBytes[0], 0u);
}

TEST_P(DecoderTest, IgnoreAll) {
  auto decoder = CreateDecoder("seine_sdr_gainmap_srgb.avif");
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode = AVIF_IMAGE_CONTENT_NONE;
  auto result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
  avifImage* decoded = decoder->image;
  ASSERT_NE(decoded, nullptr);

  EXPECT_NE(decoded->gainMap, nullptr);
  ASSERT_EQ(decoded->gainMap->image, nullptr);

  // But trying to access the next image should give an error because both
  // ignoreColorAndAlpha and enableDecodingGainMap are set.
  ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_NO_CONTENT);
}

TEST_P(DecoderTest, GainMapReadMemoryFails) {
  auto data =
      testutil::read_file(GetFilename("seine_sdr_gainmap_srgb.avif").c_str());
  ASSERT_GT(data.size(), 0);
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  avifImage image;
  // avifDecoderReadMemory does not support gainmaps.
  EXPECT_NE(
      avifDecoderReadMemory(decoder.get(), &image, data.data(), data.size()),
      AVIF_RESULT_OK);
}

// Callers bound memory allocations based on the base image dimensions, so gain
// maps that are larger than the base image in either dimension are ignored.
TEST_P(DecoderTest, GainMapLargerThanBaseImage) {
  struct TestFile {
    const char* file_name;
    uint32_t width;
    uint32_t height;
  };
  for (const TestFile& file :
       {TestFile{"gainmap_larger_than_base_wider.avif", 16, 16},
        TestFile{"gainmap_larger_than_base_taller.avif", 16, 16},
        TestFile{"gainmap_larger_than_base_wider_and_taller.avif", 16, 16},
        TestFile{"gainmap_larger_than_base_grid.avif", 128, 64}}) {
    for (avifImageContentTypeFlags image_content_to_decode :
         {AVIF_IMAGE_CONTENT_ALL, AVIF_IMAGE_CONTENT_COLOR_AND_ALPHA,
          AVIF_IMAGE_CONTENT_GAIN_MAP}) {
      SCOPED_TRACE(::testing::Message()
                   << file.file_name << " " << image_content_to_decode);
      auto decoder = CreateDecoder(file.file_name);
      ASSERT_NE(decoder, nullptr);
      decoder->imageContentToDecode = image_content_to_decode;
      auto result = avifDecoderParse(decoder.get());
      ASSERT_EQ(result, AVIF_RESULT_OK)
          << avifResultToString(result) << " " << decoder->diag.error;
      ASSERT_NE(decoder->image, nullptr);
      EXPECT_EQ(decoder->image->width, file.width);
      EXPECT_EQ(decoder->image->height, file.height);
      EXPECT_EQ(decoder->image->gainMap, nullptr);
      if (!testutil::Av1DecoderAvailable()) {
        continue;
      }
      if (image_content_to_decode == AVIF_IMAGE_CONTENT_GAIN_MAP) {
        // There is nothing to decode since the gain map is ignored.
        EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_NO_CONTENT);
      } else {
        EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
        EXPECT_EQ(decoder->image->gainMap, nullptr);
      }
    }
  }
}

TEST(DecoderTest, KeyFrame) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder("colors-animated-12bpc-keyframes-0-2-3.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);

  // The first frame is always a keyframe.
  EXPECT_TRUE(avifDecoderIsKeyframe(decoder.get(), 0));
  EXPECT_EQ(avifDecoderNearestKeyframe(decoder.get(), 0), 0);

  // The encoder may choose to use a keyframe here, even without FORCE_KEYFRAME.
  // It seems not to.
  EXPECT_FALSE(avifDecoderIsKeyframe(decoder.get(), 1));
  EXPECT_EQ(avifDecoderNearestKeyframe(decoder.get(), 1), 0);

  EXPECT_TRUE(avifDecoderIsKeyframe(decoder.get(), 2));
  EXPECT_EQ(avifDecoderNearestKeyframe(decoder.get(), 2), 2);

  // The encoder seems to prefer a keyframe here
  // (gradient too different from plain color).
  EXPECT_TRUE(avifDecoderIsKeyframe(decoder.get(), 3));
  EXPECT_EQ(avifDecoderNearestKeyframe(decoder.get(), 3), 3);

  // This is the same frame as the previous one. It should not be a keyframe.
  EXPECT_FALSE(avifDecoderIsKeyframe(decoder.get(), 4));
  EXPECT_EQ(avifDecoderNearestKeyframe(decoder.get(), 4), 3);
}

TEST_P(DecoderTest, Progressive) {
  if (codec_choice() == AVIF_CODEC_CHOICE_NATIVE) {
    GTEST_SKIP() << "The native AV1 decoder does not support progressive "
                    "(layered) images.";
  }
#ifdef __ANDROID__
  // Android MediaCodec does not support progressive images.
  GTEST_SKIP() << "Unsupported test on Android.";
#endif
  struct Params {
    const char* file_name;
    uint32_t width;
    uint32_t height;
    uint32_t layer_count;
  };
  Params params[] = {
      {"progressive/progressive_dimension_change.avif", 256, 256, 2},
      {"progressive/progressive_layered_grid.avif", 512, 256, 2},
      {"progressive/progressive_quality_change.avif", 256, 256, 2},
      {"progressive/progressive_same_layers.avif", 256, 256, 4},
      {"progressive/tiger_3layer_1res.avif", 1216, 832, 3},
      {"progressive/tiger_3layer_3res.avif", 1216, 832, 3},
  };
  for (const auto& param : params) {
    DecoderPtr decoder(NewDecoder());
    ASSERT_NE(decoder, nullptr);
    decoder->allowProgressive = true;
    ASSERT_EQ(avifDecoderSetIOFile(decoder.get(),
                                   GetFilename(param.file_name).c_str()),
              AVIF_RESULT_OK);
    ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
    EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
    ASSERT_EQ(decoder->progressiveState, AVIF_PROGRESSIVE_STATE_ACTIVE);
    ASSERT_EQ(static_cast<uint32_t>(decoder->imageCount), param.layer_count);

    for (uint32_t layer = 0; layer < param.layer_count; ++layer) {
      ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
      // libavif scales frame automatically.
      ASSERT_EQ(decoder->image->width, param.width);
      ASSERT_EQ(decoder->image->height, param.height);
    }
  }
}

// A test for https://github.com/AOMediaCodec/libavif/issues/1086 to prevent
// regression.
TEST(DecoderTest, ParseICC) {
  auto decoder = CreateDecoder("paris_icc_exif_xmp.avif");
  ASSERT_NE(decoder, nullptr);

  decoder->ignoreXMP = AVIF_TRUE;
  decoder->ignoreExif = AVIF_TRUE;
  EXPECT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);

  ASSERT_GE(decoder->image->icc.size, 4u);
  EXPECT_EQ(decoder->image->icc.data[0], 0);
  EXPECT_EQ(decoder->image->icc.data[1], 0);
  EXPECT_EQ(decoder->image->icc.data[2], 2);
  EXPECT_EQ(decoder->image->icc.data[3], 84);

  ASSERT_EQ(decoder->image->exif.size, 0u);
  ASSERT_EQ(decoder->image->xmp.size, 0u);

  decoder->ignoreXMP = AVIF_FALSE;
  decoder->ignoreExif = AVIF_FALSE;
  EXPECT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);

  ASSERT_GE(decoder->image->exif.size, 4u);
  EXPECT_EQ(decoder->image->exif.data[0], 73);
  EXPECT_EQ(decoder->image->exif.data[1], 73);
  EXPECT_EQ(decoder->image->exif.data[2], 42);
  EXPECT_EQ(decoder->image->exif.data[3], 0);

  ASSERT_GE(decoder->image->xmp.size, 4u);
  EXPECT_EQ(decoder->image->xmp.data[0], 60);
  EXPECT_EQ(decoder->image->xmp.data[1], 63);
  EXPECT_EQ(decoder->image->xmp.data[2], 120);
  EXPECT_EQ(decoder->image->xmp.data[3], 112);
}

TEST(DecoderTest, ParseExifNonZeroTiffOffset) {
  auto decoder = CreateDecoder("paris_exif_non_zero_tiff_offset.avif");
  ASSERT_NE(decoder, nullptr);

  EXPECT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);

  ASSERT_EQ(decoder->image->exif.size, 1129);
  EXPECT_EQ(decoder->image->exif.data[0], 0);
  EXPECT_EQ(decoder->image->exif.data[1], 0);
  EXPECT_EQ(decoder->image->exif.data[2], 0);
  EXPECT_EQ(decoder->image->exif.data[3], 73);
  EXPECT_EQ(decoder->image->exif.data[4], 73);
  EXPECT_EQ(decoder->image->exif.data[5], 42);
  EXPECT_EQ(decoder->image->exif.data[6], 0);
}

bool CompareImages(const avifImage& image1, const avifImage image2) {
  EXPECT_EQ(image1.width, image2.width);
  EXPECT_EQ(image1.height, image2.height);
  EXPECT_EQ(image1.depth, image2.depth);
  EXPECT_EQ(image1.yuvFormat, image2.yuvFormat);
  EXPECT_EQ(image1.yuvRange, image2.yuvRange);
  for (int c = 0; c < 4; ++c) {
    const uint8_t* row1 = avifImagePlane(&image1, c);
    const uint8_t* row2 = avifImagePlane(&image2, c);
    if (!row1 != !row2) {
      return false;
    }
    const uint32_t row_bytes1 = avifImagePlaneRowBytes(&image1, c);
    const uint32_t row_bytes2 = avifImagePlaneRowBytes(&image2, c);
    const uint32_t plane_width = avifImagePlaneWidth(&image1, c);
    const uint32_t plane_height = avifImagePlaneHeight(&image1, c);
    for (uint32_t y = 0; y < plane_height; ++y) {
      if (avifImageUsesU16(&image1)) {
        if (!std::equal(reinterpret_cast<const uint16_t*>(row1),
                        reinterpret_cast<const uint16_t*>(row1) + plane_width,
                        reinterpret_cast<const uint16_t*>(row2))) {
          return false;
        }
      } else {
        if (!std::equal(row1, row1 + plane_width, row2)) {
          return false;
        }
      }
      row1 += row_bytes1;
      row2 += row_bytes2;
    }
  }
  return true;
}

class ImageCopyFileTest : public testing::TestWithParam<const char*> {};

TEST_P(ImageCopyFileTest, ImageCopy) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder(GetParam());
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);

  ImagePtr image2(avifImageCreateEmpty());
  ASSERT_EQ(avifImageCopy(image2.get(), decoder->image, AVIF_PLANES_ALL),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *image2));
}

INSTANTIATE_TEST_SUITE_P(ImageCopyFileTestInstance, ImageCopyFileTest,
                         testing::ValuesIn({"paris_10bpc.avif", "alpha.avif",
                                            "colors-animated-8bpc.avif"}));

class ImageCopyTest : public testing::TestWithParam<
                          std::tuple<int, avifPixelFormat, avifPlanesFlag>> {};

TEST_P(ImageCopyTest, RightEdgeDoesNotOverreadInLastRow) {
  const auto depth = std::get<0>(GetParam());
  const auto pixel_format = std::get<1>(GetParam());

  if ((pixel_format == AVIF_PIXEL_FORMAT_ANDROID_P010 && depth == 8) ||
      ((pixel_format == AVIF_PIXEL_FORMAT_ANDROID_NV12 ||
        pixel_format == AVIF_PIXEL_FORMAT_ANDROID_NV21) &&
       depth != 8)) {
    GTEST_SKIP() << "This combination of parameters is not valid. Skipping.";
  }

  constexpr int kWidth = 100;
  constexpr int kHeight = 100;
  ImagePtr src(avifImageCreate(kWidth, kHeight, depth, pixel_format));

  const auto planes = std::get<2>(GetParam());
  ASSERT_EQ(avifImageAllocatePlanes(src.get(), planes), AVIF_RESULT_OK);
  for (int i = 0; i < 4; ++i) {
    const int plane_width_bytes =
        avifImagePlaneWidth(src.get(), i) * ((depth > 8) ? 2 : 1);
    const int plane_height = avifImagePlaneHeight(src.get(), i);
    uint8_t* plane = avifImagePlane(src.get(), i);
    const int row_bytes = avifImagePlaneRowBytes(src.get(), i);
    for (int y = 0; y < plane_height; ++y) {
      std::iota(plane, plane + plane_width_bytes, y);
      plane += row_bytes;
    }
  }

  constexpr int kSubsetWidth = 20;
  constexpr int kSubsetHeight = kHeight;

  // Get a subset of the image near the right edge (last 20 pixel columns). If
  // the copy implementation is correct, it will copy the exact 20 columns
  // without over-reading beyond the |width| pixels irrespective of what the
  // source stride is.
  ImagePtr subset_image(avifImageCreateEmpty());
  const avifCropRect rect{
      .x = 80, .y = 0, .width = kSubsetWidth, .height = kSubsetHeight};
  auto result = avifImageSetViewRect(subset_image.get(), src.get(), &rect);
  ASSERT_EQ(result, AVIF_RESULT_OK);
  auto* image = subset_image.get();

  EXPECT_EQ(image->width, kSubsetWidth);
  EXPECT_EQ(image->height, kSubsetHeight);

  // Perform a copy of the subset.
  ImagePtr copied_image(avifImageCreateEmpty());
  result =
      avifImageCopy(copied_image.get(), subset_image.get(), AVIF_PLANES_ALL);
  ASSERT_EQ(result, AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*subset_image, *copied_image));
}

INSTANTIATE_TEST_SUITE_P(
    ImageCopyTestInstance, ImageCopyTest,
    testing::Combine(testing::ValuesIn({8, 10, 12}),
                     testing::ValuesIn({AVIF_PIXEL_FORMAT_YUV420,
                                        AVIF_PIXEL_FORMAT_ANDROID_NV12,
                                        AVIF_PIXEL_FORMAT_ANDROID_NV21,
                                        AVIF_PIXEL_FORMAT_ANDROID_P010}),
                     testing::ValuesIn({AVIF_PLANES_ALL, AVIF_PLANES_YUV})));

class ImageSetViewRectTest
    : public testing::TestWithParam<
          std::tuple<avifPixelFormat, std::tuple<avifCropRect, bool>>> {};

TEST_P(ImageSetViewRectTest, SetViewRect) {
  const auto pixel_format = std::get<0>(GetParam());

  constexpr int kWidth = 100;
  constexpr int kHeight = 100;
  const int depth =
      (pixel_format == crabbyavif::AVIF_PIXEL_FORMAT_ANDROID_P010) ? 10 : 8;
  ImagePtr src(avifImageCreate(kWidth, kHeight, depth, pixel_format));

  ASSERT_EQ(avifImageAllocatePlanes(src.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);
  for (int i = 0; i < 4; ++i) {
    const int plane_width_bytes =
        avifImagePlaneWidth(src.get(), i) * ((depth > 8) ? 2 : 1);
    const int plane_height = avifImagePlaneHeight(src.get(), i);
    uint8_t* plane = avifImagePlane(src.get(), i);
    const int row_bytes = avifImagePlaneRowBytes(src.get(), i);
    for (int y = 0; y < plane_height; ++y) {
      std::iota(plane, plane + plane_width_bytes, y);
      plane += row_bytes;
    }
  }

  const auto rect = std::get<0>(std::get<1>(GetParam()));
  const auto expect_success = std::get<1>(std::get<1>(GetParam()));

  ImagePtr sub_image(avifImageCreateEmpty());
  auto result = avifImageSetViewRect(sub_image.get(), src.get(), &rect);
  if (expect_success) {
    ASSERT_EQ(result, AVIF_RESULT_OK);
    EXPECT_EQ(sub_image->width, rect.width);
    EXPECT_EQ(sub_image->height, rect.height);
  } else {
    EXPECT_NE(result, AVIF_RESULT_OK);
  }
}

#ifdef __ANDROID__
static constexpr bool kTrueOnlyOnAndroid = true;
#else
static constexpr bool kTrueOnlyOnAndroid = false;
#endif

static const std::tuple<avifCropRect, bool> kRects[] = {
    // All possible even/odd combinations.
    {{.x = 60, .y = 60, .width = 20, .height = 20}, true},
    {{.x = 60, .y = 60, .width = 20, .height = 21}, true},
    {{.x = 60, .y = 60, .width = 21, .height = 20}, true},
    {{.x = 60, .y = 60, .width = 21, .height = 21}, true},
    {{.x = 60, .y = 61, .width = 20, .height = 20}, kTrueOnlyOnAndroid},
    {{.x = 60, .y = 61, .width = 20, .height = 21}, kTrueOnlyOnAndroid},
    {{.x = 60, .y = 61, .width = 21, .height = 20}, kTrueOnlyOnAndroid},
    {{.x = 60, .y = 61, .width = 21, .height = 21}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 60, .width = 20, .height = 20}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 60, .width = 20, .height = 21}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 60, .width = 21, .height = 20}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 60, .width = 21, .height = 21}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 61, .width = 20, .height = 20}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 61, .width = 20, .height = 21}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 61, .width = 21, .height = 20}, kTrueOnlyOnAndroid},
    {{.x = 61, .y = 61, .width = 21, .height = 21}, kTrueOnlyOnAndroid},
    // Out of bounds.
    {{.x = 100, .y = 60, .width = 20, .height = 20}, false},
    {{.x = 60, .y = 100, .width = 20, .height = 20}, false},
    {{.x = 99, .y = 60, .width = 2, .height = 20}, false},
    {{.x = 60, .y = 99, .width = 20, .height = 2}, false},
};

INSTANTIATE_TEST_SUITE_P(
    ImageSetViewRectTestInstance, ImageSetViewRectTest,
    testing::Combine(testing::ValuesIn({AVIF_PIXEL_FORMAT_YUV420,
                                        AVIF_PIXEL_FORMAT_ANDROID_NV12,
                                        AVIF_PIXEL_FORMAT_ANDROID_P010}),
                     testing::ValuesIn(kRects)));

TEST_P(DecoderTest, SetRawIO) {
  if (codec_choice() == AVIF_CODEC_CHOICE_NATIVE) {
    GTEST_SKIP() << "The native AV1 decoder does not support inter frames.";
  }
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  auto data =
      testutil::read_file(GetFilename("colors-animated-8bpc.avif").c_str());
  ASSERT_EQ(avifDecoderSetIOMemory(decoder.get(), data.data(), data.size()),
            AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_FALSE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageCount, 5);
  EXPECT_EQ(decoder->repetitionCount, 0);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  }

  decoder.reset(NewDecoder());
  EXPECT_NE(avifDecoderSetIOMemory(decoder.get(), nullptr, 0), AVIF_RESULT_OK);
}

TEST_P(DecoderTest, SetCustomIO) {
  if (codec_choice() == AVIF_CODEC_CHOICE_NATIVE) {
    GTEST_SKIP() << "The native AV1 decoder does not support inter frames.";
  }
  auto data =
      testutil::read_file(GetFilename("colors-animated-8bpc.avif").c_str());
  avifROData ro_data = {.data = data.data(), .size = data.size()};
  avifIO io = {.destroy = nullptr,
               .read = io_read,
               .sizeHint = data.size(),
               .persistent = false,
               .data = &ro_data};
  // |io| must outlive the decoder.
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  avifDecoderSetIO(decoder.get(), &io);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_FALSE);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageCount, 5);
  EXPECT_EQ(decoder->repetitionCount, 0);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  }
}

struct NonPersistentIO {
  std::vector<uint8_t> data;
  avifROData ro_data;
};

// Every read call will explicitly invalidate the pointer returned by the
// previous read call to ensure non-persistent IO.
avifResult non_persistent_read(struct avifIO* io, uint32_t flags,
                               uint64_t offset, size_t size, avifROData* out) {
  NonPersistentIO* io_data = reinterpret_cast<NonPersistentIO*>(io->data);
  if (flags != 0 || offset > io_data->ro_data.size) {
    return AVIF_RESULT_IO_ERROR;
  }
  uint64_t available_size = io_data->ro_data.size - offset;
  if (size > available_size) {
    size = static_cast<size_t>(available_size);
  }
  // Clear the existing vector.
  std::vector<uint8_t>().swap(io_data->data);
  // Copy new data into the vector.
  io_data->data.reserve(size);
  io_data->data.assign(io_data->ro_data.data + offset,
                       io_data->ro_data.data + offset + size);
  // Set the output.
  out->data = io_data->data.data();
  out->size = size;
  return AVIF_RESULT_OK;
}

TEST_P(DecoderTest, NonPersistentIOBug506387278) {
  if (codec_choice() == AVIF_CODEC_CHOICE_NATIVE) {
    GTEST_SKIP() << "The native AV1 decoder does not support inter frames.";
  }
#ifdef __ANDROID__
  // Android MediaCodec does not support decoding this constructed poc file.
  GTEST_SKIP() << "Unsupported test on Android.";
#endif
  auto data = testutil::read_file(GetFilename("poc_b_506387278.avif").c_str());
  NonPersistentIO io_data;
  io_data.ro_data = {.data = data.data(), .size = data.size()};
  avifIO io = {.destroy = nullptr,
               .read = non_persistent_read,
               .sizeHint = data.size(),
               .persistent = false,
               .data = &io_data};
  // |io| must outlive the decoder.
  DecoderPtr decoder(NewDecoder());
  ASSERT_NE(decoder, nullptr);
  avifDecoderSetIO(decoder.get(), &io);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->imageSequenceTrackPresent, AVIF_TRUE);
  EXPECT_EQ(decoder->imageCount, 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  }
}

TEST(DecoderTest, IOMemoryReader) {
  auto data =
      testutil::read_file(GetFilename("colors-animated-8bpc.avif").c_str());
  avifIO* io = avifIOCreateMemoryReader(data.data(), data.size());
  ASSERT_NE(io, nullptr);
  EXPECT_EQ(io->sizeHint, data.size());
  avifROData ro_data;
  // Read 10 bytes from the beginning.
  io->read(io, 0, 0, 10, &ro_data);
  EXPECT_EQ(ro_data.size, 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(ro_data.data[i], data[i]);
  }
  // Read 10 bytes from the middle.
  io->read(io, 0, 50, 10, &ro_data);
  EXPECT_EQ(ro_data.size, 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(ro_data.data[i], data[i + 50]);
  }
  avifIODestroy(io);
}

TEST(DecoderTest, IOFileReader) {
  const char* file_name = "colors-animated-8bpc.avif";
  auto data = testutil::read_file(GetFilename(file_name).c_str());
  avifIO* io = avifIOCreateFileReader(GetFilename(file_name).c_str());
  ASSERT_NE(io, nullptr);
  EXPECT_EQ(io->sizeHint, data.size());
  avifROData ro_data;
  // Read 10 bytes from the beginning.
  io->read(io, 0, 0, 10, &ro_data);
  EXPECT_EQ(ro_data.size, 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(ro_data.data[i], data[i]);
  }
  // Read 10 bytes from the middle.
  io->read(io, 0, 50, 10, &ro_data);
  EXPECT_EQ(ro_data.size, 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(ro_data.data[i], data[i + 50]);
  }
  avifIODestroy(io);
}

TEST(DecoderTest, IODestroyCustomIOWithoutDestroyCallback) {
  // avifIODestroy() only calls io->destroy if it is set, so it must be a no-op
  // for a caller-owned avifIO without a destroy callback (it must not free the
  // avifIO or its data).
  avifROData ro_data = {.data = nullptr, .size = 0};
  avifIO io = {.destroy = nullptr,
               .read = io_read,
               .sizeHint = 0,
               .persistent = false,
               .data = &ro_data};
  avifIODestroy(&io);
  EXPECT_EQ(io.data, &ro_data);
}

void CountingIODestroy(avifIO* io) {
  ++*reinterpret_cast<int*>(io->data);
  delete io;
}

TEST(DecoderTest, IODestroyCallsCustomDestroyCallback) {
  // avifIODestroy() must call the avifIO's destroy callback when it is set.
  int destroy_count = 0;
  avifIO* io = new avifIO{.destroy = CountingIODestroy,
                          .read = io_read,
                          .sizeHint = 0,
                          .persistent = false,
                          .data = &destroy_count};
  avifIODestroy(io);
  EXPECT_EQ(destroy_count, 1);
}

TEST(DecoderTest, DecoderOwnsMemoryReader) {
  // avifDecoderSetIO() transfers ownership of the avifIO to the decoder, which
  // must release it (checked by the leak sanitizer).
  auto data =
      testutil::read_file(GetFilename("colors-animated-8bpc.avif").c_str());
  avifIO* io = avifIOCreateMemoryReader(data.data(), data.size());
  ASSERT_NE(io, nullptr);
  DecoderPtr decoder(avifDecoderCreate());
  ASSERT_NE(decoder, nullptr);
  avifDecoderSetIO(decoder.get(), io);
  EXPECT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
}

class ScaleTest : public testing::TestWithParam<const char*> {};

TEST_P(ScaleTest, Scaling) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder(GetParam());
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);

  const uint32_t scaled_width =
      static_cast<uint32_t>(decoder->image->width * 0.8);
  const uint32_t scaled_height =
      static_cast<uint32_t>(decoder->image->height * 0.8);

  ASSERT_EQ(
      avifImageScale(decoder->image, scaled_width, scaled_height, nullptr),
      AVIF_RESULT_OK);
  EXPECT_EQ(decoder->image->width, scaled_width);
  EXPECT_EQ(decoder->image->height, scaled_height);

  // Scaling to a larger dimension is not supported.
  EXPECT_NE(avifImageScale(decoder->image, decoder->image->width * 2,
                           decoder->image->height * 0.5, nullptr),
            AVIF_RESULT_OK);
  EXPECT_NE(avifImageScale(decoder->image, decoder->image->width * 0.5,
                           decoder->image->height * 2, nullptr),
            AVIF_RESULT_OK);
  EXPECT_NE(avifImageScale(decoder->image, decoder->image->width * 2,
                           decoder->image->height * 2, nullptr),
            AVIF_RESULT_OK);
}

INSTANTIATE_TEST_SUITE_P(ScaleTestInstance, ScaleTest,
                         testing::ValuesIn({"paris_10bpc.avif",
                                            "paris_icc_exif_xmp.avif"}));

TEST(ScaleTest, ScaleP010) {
  const int width = 100;
  const int height = 50;
  ImagePtr image(
      avifImageCreate(width, height, 10, AVIF_PIXEL_FORMAT_ANDROID_P010));
  ASSERT_EQ(avifImageAllocatePlanes(image.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);

  const uint32_t scaled_width = static_cast<uint32_t>(width * 0.8);
  const uint32_t scaled_height = static_cast<uint32_t>(height * 0.6);

  ASSERT_EQ(avifImageScale(image.get(), scaled_width, scaled_height, nullptr),
            AVIF_RESULT_OK);
  EXPECT_EQ(image->width, scaled_width);
  EXPECT_EQ(image->height, scaled_height);
  EXPECT_EQ(image->depth, 10);
  // When scaling a P010 image, crabbyavif converts it into an I010 (Yuv420)
  // image.
  EXPECT_EQ(image->yuvFormat, AVIF_PIXEL_FORMAT_YUV420);
  for (int c = 0; c < 3; ++c) {
    EXPECT_NE(image->yuvPlanes[c], nullptr);
    EXPECT_GT(image->yuvRowBytes[c], 0);
  }
  EXPECT_NE(image->alphaPlane, nullptr);
  EXPECT_NE(image->alphaRowBytes, 0);
}

TEST(ScaleTest, ScaleNV12OddDimensions) {
  const int width = 99;
  const int height = 49;
  ImagePtr image(
      avifImageCreate(width, height, 8, AVIF_PIXEL_FORMAT_ANDROID_NV12));
  ASSERT_EQ(avifImageAllocatePlanes(image.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);

  const uint32_t scaled_width = 49;
  const uint32_t scaled_height = 24;

  ASSERT_EQ(avifImageScale(image.get(), scaled_width, scaled_height, nullptr),
            AVIF_RESULT_OK);
  EXPECT_EQ(image->width, scaled_width);
  EXPECT_EQ(image->height, scaled_height);
  EXPECT_EQ(image->depth, 8);
  EXPECT_EQ(image->yuvFormat, AVIF_PIXEL_FORMAT_ANDROID_NV12);
  for (int c = 0; c < 2; ++c) {
    EXPECT_NE(image->yuvPlanes[c], nullptr);
    EXPECT_GT(image->yuvRowBytes[c], 0);
  }
  EXPECT_EQ(image->yuvPlanes[2], nullptr);
  EXPECT_EQ(image->yuvRowBytes[2], 0);
  EXPECT_NE(image->alphaPlane, nullptr);
  EXPECT_NE(image->alphaRowBytes, 0);
}

TEST(ScaleTest, ScaleNV12WithCopyOddDimensions) {
  const int width = 99;
  const int height = 49;
  ImagePtr image(
      avifImageCreate(width, height, 8, AVIF_PIXEL_FORMAT_ANDROID_NV12));
  ASSERT_EQ(avifImageAllocatePlanes(image.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);

  // Create a copy of the image and scale the copy (this mimic's skia's
  // implementation).
  ImagePtr image2(avifImageCreateEmpty());
  ASSERT_EQ(avifImageCopy(image2.get(), image.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);

  const uint32_t scaled_width = 49;
  const uint32_t scaled_height = 24;

  ASSERT_EQ(avifImageScale(image2.get(), scaled_width, scaled_height, nullptr),
            AVIF_RESULT_OK);
  EXPECT_EQ(image2->width, scaled_width);
  EXPECT_EQ(image2->height, scaled_height);
  EXPECT_EQ(image2->depth, 8);
  EXPECT_EQ(image2->yuvFormat, AVIF_PIXEL_FORMAT_ANDROID_NV12);
  for (int c = 0; c < 2; ++c) {
    EXPECT_NE(image->yuvPlanes[c], nullptr);
    EXPECT_GT(image->yuvRowBytes[c], 0);
  }
  EXPECT_EQ(image->yuvPlanes[2], nullptr);
  EXPECT_EQ(image->yuvRowBytes[2], 0);
  EXPECT_NE(image->alphaPlane, nullptr);
  EXPECT_NE(image->alphaRowBytes, 0);
}

struct InvalidClapPropertyParam {
  uint32_t width;
  uint32_t height;
  avifPixelFormat yuv_format;
  avifCleanApertureBox clap;
};

constexpr InvalidClapPropertyParam kInvalidClapPropertyTestParams[] = {
    // Zero or negative denominators.
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {96, 0, 132, 1, 0, 1, 0, 1}},
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {96, static_cast<uint32_t>(-1), 132, 1, 0, 1, 0, 1}},
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {96, 1, 132, 0, 0, 1, 0, 1}},
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {96, 1, 132, static_cast<uint32_t>(-1), 0, 1, 0, 1}},
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {96, 1, 132, 1, 0, 0, 0, 1}},
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {96, 1, 132, 1, 0, static_cast<uint32_t>(-1), 0, 1}},
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {96, 1, 132, 1, 0, 1, 0, 0}},
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {96, 1, 132, 1, 0, 1, 0, static_cast<uint32_t>(-1)}},
    // Zero or negative clean aperture width or height.
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {static_cast<uint32_t>(-96), 1, 132, 1, 0, 1, 0, 1}},
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {0, 1, 132, 1, 0, 1, 0, 1}},
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {96, 1, static_cast<uint32_t>(-132), 1, 0, 1, 0, 1}},
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {96, 1, 0, 1, 0, 1, 0, 1}},
    // Clean aperture width or height is not an integer.
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {96, 5, 132, 1, 0, 1, 0, 1}},
    {120, 160, AVIF_PIXEL_FORMAT_YUV420, {96, 1, 132, 5, 0, 1, 0, 1}},
    // pcX = 103 + (722 - 1)/2 = 463.5
    // pcY = -308 + (1024 - 1)/2 = 203.5
    // leftmost = 463.5 - (385 - 1)/2 = 271.5 (not an integer)
    // topmost = 203.5 - (330 - 1)/2 = 39
    {722,
     1024,
     AVIF_PIXEL_FORMAT_YUV420,
     {385, 1, 330, 1, 103, 1, static_cast<uint32_t>(-308), 1}},
    // pcX = -308 + (1024 - 1)/2 = 203.5
    // pcY = 103 + (722 - 1)/2 = 463.5
    // leftmost = 203.5 - (330 - 1)/2 = 39
    // topmost = 463.5 - (385 - 1)/2 = 271.5 (not an integer)
    {1024,
     722,
     AVIF_PIXEL_FORMAT_YUV420,
     {330, 1, 385, 1, static_cast<uint32_t>(-308), 1, 103, 1}},
    // pcX = -1/2 + (99 - 1)/2 = 48.5
    // pcY = -1/2 + (99 - 1)/2 = 48.5
    // leftmost = 48.5 - (99 - 1)/2 = -0.5 (not an integer)
    // topmost = 48.5 - (99 - 1)/2 = -0.5 (not an integer)
    {99,
     99,
     AVIF_PIXEL_FORMAT_YUV420,
     {99, 1, 99, 1, static_cast<uint32_t>(-1), 2, static_cast<uint32_t>(-1),
      2}},
};

using InvalidClapPropertyTest =
    ::testing::TestWithParam<InvalidClapPropertyParam>;

// Negative tests for the avifCropRectConvertCleanApertureBox() function.
TEST_P(InvalidClapPropertyTest, ValidateClapProperty) {
  const InvalidClapPropertyParam& param = GetParam();
  avifCropRect crop_rect;
  avifDiagnostics diag;
  EXPECT_FALSE(avifCropRectConvertCleanApertureBox(&crop_rect, &param.clap,
                                                   param.width, param.height,
                                                   param.yuv_format, &diag));
}

INSTANTIATE_TEST_SUITE_P(Parameterized, InvalidClapPropertyTest,
                         ::testing::ValuesIn(kInvalidClapPropertyTestParams));

struct ValidClapPropertyParam {
  uint32_t width;
  uint32_t height;
  avifPixelFormat yuv_format;
  avifCleanApertureBox clap;

  avifCropRect expected_crop_rect;
};

constexpr ValidClapPropertyParam kValidClapPropertyTestParams[] = {
    // pcX = 0 + (120 - 1)/2 = 59.5
    // pcY = 0 + (160 - 1)/2 = 79.5
    // leftmost = 59.5 - (96 - 1)/2 = 12
    // topmost = 79.5 - (132 - 1)/2 = 14
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {96, 1, 132, 1, 0, 1, 0, 1},
     {12, 14, 96, 132}},
    // pcX = -30 + (120 - 1)/2 = 29.5
    // pcY = -40 + (160 - 1)/2 = 39.5
    // leftmost = 29.5 - (60 - 1)/2 = 0
    // topmost = 39.5 - (80 - 1)/2 = 0
    {120,
     160,
     AVIF_PIXEL_FORMAT_YUV420,
     {60, 1, 80, 1, static_cast<uint32_t>(-30), 1, static_cast<uint32_t>(-40),
      1},
     {0, 0, 60, 80}},
    // pcX = -1/2 + (100 - 1)/2 = 49
    // pcY = -1/2 + (100 - 1)/2 = 49
    // leftmost = 49 - (99 - 1)/2 = 0
    // topmost = 49 - (99 - 1)/2 = 0
    {100,
     100,
     AVIF_PIXEL_FORMAT_YUV420,
     {99, 1, 99, 1, static_cast<uint32_t>(-1), 2, static_cast<uint32_t>(-1), 2},
     {0, 0, 99, 99}},
};

using ValidClapPropertyTest = ::testing::TestWithParam<ValidClapPropertyParam>;

// Positive tests for the avifCropRectConvertCleanApertureBox() function.
TEST_P(ValidClapPropertyTest, ValidateClapProperty) {
  const ValidClapPropertyParam& param = GetParam();
  avifCropRect crop_rect;
  avifDiagnostics diag;
  EXPECT_TRUE(avifCropRectConvertCleanApertureBox(&crop_rect, &param.clap,
                                                  param.width, param.height,
                                                  param.yuv_format, &diag))
      << diag.error;
  EXPECT_EQ(crop_rect.x, param.expected_crop_rect.x);
  EXPECT_EQ(crop_rect.y, param.expected_crop_rect.y);
  EXPECT_EQ(crop_rect.width, param.expected_crop_rect.width);
  EXPECT_EQ(crop_rect.height, param.expected_crop_rect.height);
}

INSTANTIATE_TEST_SUITE_P(Parameterized, ValidClapPropertyTest,
                         ::testing::ValuesIn(kValidClapPropertyTestParams));

TEST(DecoderTest, ClapIrotImirNonEssential) {
  // Invalid file with non-essential transformative properties.
  auto decoder = CreateDecoder("clap_irot_imir_non_essential.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_BMFF_PARSE_FAILED);
}

TEST(DecoderTest, PeekCompatibleFileType) {
  EXPECT_EQ(avifPeekCompatibleFileType(nullptr), AVIF_FALSE);

  constexpr static uint8_t data[] = {1, 2, 3, 4};
  avifROData input = {nullptr, 0};
  EXPECT_EQ(avifPeekCompatibleFileType(&input), AVIF_FALSE);

  input.data = data;
  EXPECT_EQ(avifPeekCompatibleFileType(&input), AVIF_FALSE);
}

TEST(DecoderTest, NullCases) {
  avifIO io;
  auto decoder = CreateDecoder("clap_irot_imir_non_essential.avif");
  avifDecoderSetIO(nullptr, &io);
  avifDecoderSetIO(decoder.get(), nullptr);
  avifDecoderSetIO(nullptr, nullptr);

  EXPECT_NE(avifDecoderSetIOFile(nullptr, "something.avif"), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderSetIOFile(decoder.get(), nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderSetIOFile(nullptr, nullptr), AVIF_RESULT_OK);

  uint8_t raw_data[10] = {0};
  EXPECT_NE(avifDecoderSetIOMemory(nullptr, nullptr, 0), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderSetIOMemory(nullptr, raw_data, 10), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderSetIOMemory(decoder.get(), nullptr, 0), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderSetIOMemory(decoder.get(), nullptr, 10), AVIF_RESULT_OK);

  EXPECT_NE(avifDecoderSetSource(nullptr, AVIF_DECODER_SOURCE_AUTO),
            AVIF_RESULT_OK);

  EXPECT_NE(avifDecoderParse(nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNextImage(nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNthImage(nullptr, 0), AVIF_RESULT_OK);
  avifDecodeOptions decode_options{.enableCropRect = AVIF_TRUE,
                                   .cropRect = {0, 0, 1, 1}};
  EXPECT_NE(avifDecoderNextImageWithOptions(nullptr, nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNextImageWithOptions(nullptr, &decode_options),
            AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNthImageWithOptions(nullptr, 0, nullptr),
            AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNthImageWithOptions(nullptr, 0, &decode_options),
            AVIF_RESULT_OK);

  avifImageTiming timing;
  EXPECT_NE(avifDecoderNthImageTiming(nullptr, 0, &timing), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNthImageTiming(decoder.get(), 0, nullptr),
            AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNthImageTiming(nullptr, 0, nullptr), AVIF_RESULT_OK);

  avifDecoderDestroy(nullptr);

  ImagePtr image(avifImageCreateEmpty());
  EXPECT_NE(avifDecoderRead(nullptr, nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderRead(decoder.get(), nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderRead(nullptr, image.get()), AVIF_RESULT_OK);

  EXPECT_NE(avifDecoderReadMemory(nullptr, nullptr, nullptr, 0),
            AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderReadMemory(decoder.get(), nullptr, nullptr, 0),
            AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderReadMemory(decoder.get(), image.get(), nullptr, 0),
            AVIF_RESULT_OK);

  EXPECT_NE(avifDecoderReadFile(nullptr, nullptr, nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderReadFile(nullptr, nullptr, "something.avif"),
            AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderReadFile(decoder.get(), nullptr, nullptr),
            AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderReadFile(decoder.get(), image.get(), nullptr),
            AVIF_RESULT_OK);

  EXPECT_NE(avifDecoderIsKeyframe(nullptr, 0), AVIF_TRUE);
  // The return value does not matter.
  avifDecoderNearestKeyframe(nullptr, 0);
  // The return value does not matter.
  avifDecoderDecodedRowCount(nullptr);

  EXPECT_NE(avifDecoderNthImageMaxExtent(nullptr, 0, nullptr), AVIF_RESULT_OK);
  EXPECT_NE(avifDecoderNthImageMaxExtent(decoder.get(), 0, nullptr),
            AVIF_RESULT_OK);
  avifExtent extent;
  EXPECT_NE(avifDecoderNthImageMaxExtent(nullptr, 0, &extent), AVIF_RESULT_OK);

  EXPECT_NE(avifDecoderReset(nullptr), AVIF_RESULT_OK);
}

struct StrictnessTestParams {
  const char* filename;
  avifStrictFlags strict_flag;
};

using StrictnessTest = ::testing::TestWithParam<StrictnessTestParams>;

TEST_P(StrictnessTest, StrictnessFlagDisabledAllowsDecode) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const StrictnessTestParams& param = GetParam();
  auto decoder = CreateDecoder(param.filename);
  ASSERT_NE(decoder, nullptr);
  // By default, non-strict files are refused.
  EXPECT_EQ(decoder->strictFlags,
            static_cast<avifStrictFlags>(AVIF_STRICT_ENABLED));
  ASSERT_NE(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  // Allow this kind of file specifically.
  decoder->strictFlags =
      static_cast<avifStrictFlags>(AVIF_STRICT_ENABLED) & ~param.strict_flag;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
}

INSTANTIATE_TEST_SUITE_P(
    DecoderTest, StrictnessTest,
    ::testing::Values(
        StrictnessTestParams{
            "white_2x2_multiple_iloc_entries_for_same_item.avif",
            AVIF_STRICT_MULTIPLE_ILOC_ENTRIES_FOR_SAME_ITEM_DISALLOWED},
        StrictnessTestParams{"white_2x2_invalid_exif.avif",
                             AVIF_STRICT_EXIF_VALID},
        StrictnessTestParams{"white_iden_irot_unordered_ipma.avif",
                             AVIF_STRICT_ITEM_IDS_INCREASING},
        StrictnessTestParams{"clap_irot_imir_non_essential.avif",
                             AVIF_STRICT_ESSENTIAL_TRANSFORMATIVE_PROPERTIES}));

struct SatoTestParams {
  const char* filename;
  bool allow_sample_transform;
  bool reset_decoder;
  uint32_t expected_depth;
};

using SatoTest =
    ::testing::TestWithParam<std::tuple<SatoTestParams, avifCodecChoice>>;

TEST_P(SatoTest, SampleTransform) {
  const SatoTestParams& param = std::get<0>(GetParam());
  const std::string path = GetFilename(param.filename);
  DecoderPtr decoder(NewDecoder(std::get<1>(GetParam())));
  ASSERT_NE(decoder, nullptr);
  decoder->allowSampleTransform =
      static_cast<crabbyavif::avifBool>(param.allow_sample_transform);
  ASSERT_EQ(avifDecoderSetIOFile(decoder.get(), path.c_str()), AVIF_RESULT_OK);

  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  if (param.reset_decoder) {
    ASSERT_EQ(avifDecoderReset(decoder.get()), AVIF_RESULT_OK);
  }
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
#ifdef __ANDROID__
  if (decoder->allowSampleTransform) {
    // Android MediaCodec does not support re-use of the same decoder instance
    // for multiple images, which is the case with Sample Transforms.
    ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_UNKNOWN_ERROR);
  } else if (param.expected_depth > 10) {
    // Android MediaCodec does not support 12-bit.
    ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_UNKNOWN_ERROR);
  } else {
    // When allowSampleTransform is false, the decoder outputs a
    // backward-compatible single-frame image.
    ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
    EXPECT_EQ(decoder->image->depth, param.expected_depth);
  }
#else
  ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->image->depth, param.expected_depth);
#endif
}

INSTANTIATE_TEST_SUITE_P(
    DecoderTest, SatoTest,
    ::testing::Combine(
        ::testing::Values(
            SatoTestParams{"weld_sato_8plus8bit.avif", false, false, 8},
            SatoTestParams{"weld_sato_8plus8bit.avif", true, false, 16},
            SatoTestParams{"weld_sato_12plus4bit.avif", false, false, 12},
            SatoTestParams{"weld_sato_12plus4bit.avif", true, false, 16},
            SatoTestParams{"weld_sato_8plus8bit.avif", false, true, 8},
            SatoTestParams{"weld_sato_8plus8bit.avif", true, true, 16},
            SatoTestParams{"weld_sato_12plus4bit.avif", false, true, 12},
            SatoTestParams{"weld_sato_12plus4bit.avif", true, true, 16}),
        ::testing::ValuesIn(GetCodecChoices())));

TEST(DecoderTest, ThumbnailNotPresent) {
  auto decoder = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderSetSource(decoder.get(), AVIF_DECODER_SOURCE_THUMBNAIL),
            AVIF_RESULT_OK);
  avifResult res = avifDecoderParse(decoder.get());
  EXPECT_TRUE(res == AVIF_RESULT_NO_CONTENT ||
              res == AVIF_RESULT_MISSING_IMAGE_ITEM);
}

TEST(DecoderTest, NthImageProperties) {
  auto decoder = CreateDecoder("colors-animated-8bpc-alpha-exif-xmp.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->imageCount, 5);
  ImagePtr props(avifImageCreateEmpty());
  ASSERT_NE(props, nullptr);
  for (uint32_t i = 0; i < 5; ++i) {
    ASSERT_EQ(avifDecoderNthImageProperties(decoder.get(), i, props.get()),
              AVIF_RESULT_OK);
    EXPECT_EQ(props->width, decoder->image->width);
    EXPECT_EQ(props->height, decoder->image->height);
    EXPECT_EQ(props->depth, decoder->image->depth);
    EXPECT_EQ(props->yuvFormat, decoder->image->yuvFormat);
    EXPECT_EQ(props->icc.data, nullptr);
    EXPECT_EQ(props->icc.size, 0u);
    EXPECT_EQ(props->exif.data, nullptr);
    EXPECT_EQ(props->exif.size, 0u);
    EXPECT_EQ(props->xmp.data, nullptr);
    EXPECT_EQ(props->xmp.size, 0u);
    for (int plane = 0; plane < AVIF_PLANE_COUNT_YUV; ++plane) {
      EXPECT_EQ(props->yuvPlanes[plane], nullptr);
    }
    EXPECT_EQ(props->alphaPlane, nullptr);
  }
  EXPECT_EQ(avifDecoderNthImageProperties(decoder.get(), 5, props.get()),
            AVIF_RESULT_NO_IMAGES_REMAINING);
}

TEST(DecoderTest, NthImagePropertiesFreesExistingPlanes) {
  auto decoder = CreateDecoder("colors-animated-8bpc-alpha-exif-xmp.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  ImagePtr props(avifImageCreate(16, 16, 8, AVIF_PIXEL_FORMAT_YUV420));
  ASSERT_NE(props, nullptr);
  ASSERT_EQ(avifImageAllocatePlanes(props.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);
  ASSERT_NE(props->yuvPlanes[0], nullptr);
  ASSERT_NE(props->alphaPlane, nullptr);
  ASSERT_EQ(avifDecoderNthImageProperties(decoder.get(), 0, props.get()),
            AVIF_RESULT_OK);
  EXPECT_EQ(props->width, decoder->image->width);
  EXPECT_EQ(props->height, decoder->image->height);
  for (int plane = 0; plane < AVIF_PLANE_COUNT_YUV; ++plane) {
    EXPECT_EQ(props->yuvPlanes[plane], nullptr);
    EXPECT_EQ(props->yuvRowBytes[plane], 0u);
  }
  EXPECT_EQ(props->alphaPlane, nullptr);
  EXPECT_EQ(props->alphaRowBytes, 0u);
}

TEST(DecoderTest, ThumbnailMaxExtentGridPerTileAlpha) {
  auto decoder = CreateDecoder("thumbnails_grid_per_tile_alpha.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderSetSource(decoder.get(), AVIF_DECODER_SOURCE_THUMBNAIL),
            AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->imageCount, 2);

  avifExtent extent{};
  ASSERT_EQ(avifDecoderNthImageMaxExtent(decoder.get(), 0, &extent),
            AVIF_RESULT_OK);
  EXPECT_EQ(extent.offset, 824u);
  EXPECT_EQ(extent.size, static_cast<size_t>((1208 + 448) - 824));

  ASSERT_EQ(avifDecoderNthImageMaxExtent(decoder.get(), 1, &extent),
            AVIF_RESULT_OK);
  EXPECT_EQ(extent.offset, 1048u);
  EXPECT_EQ(extent.size, 160u);

  EXPECT_EQ(avifDecoderNthImageMaxExtent(decoder.get(), 2, &extent),
            AVIF_RESULT_NO_IMAGES_REMAINING);
}

TEST(DecoderTest, ThumbnailMaxExtentProgressiveAndLsel) {
  auto decoder = CreateDecoder("thumbnails_progressive_lsel.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderSetSource(decoder.get(), AVIF_DECODER_SOURCE_THUMBNAIL),
            AVIF_RESULT_OK);
  decoder->allowProgressive = AVIF_TRUE;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->imageCount, 2);

  avifExtent extent{};
  ASSERT_EQ(avifDecoderNthImageMaxExtent(decoder.get(), 0, &extent),
            AVIF_RESULT_OK);
  EXPECT_EQ(extent.offset, 462u);
  EXPECT_EQ(extent.size, 58u + 302u);

  ASSERT_EQ(avifDecoderNthImageMaxExtent(decoder.get(), 1, &extent),
            AVIF_RESULT_OK);
  EXPECT_EQ(extent.offset, 462u);
  EXPECT_EQ(extent.size, 58u);

  EXPECT_EQ(avifDecoderNthImageMaxExtent(decoder.get(), 2, &extent),
            AVIF_RESULT_NO_IMAGES_REMAINING);
}

struct NativeDecodableParam {
  const char* file_name;
  avifDecoderSource source;
  avifBool allow_progressive;
  bool native_decodable;
};

class NativeDecodableTest
    : public testing::TestWithParam<NativeDecodableParam> {};

TEST_P(NativeDecodableTest, ParseAndDecode) {
  const NativeDecodableParam& param = GetParam();
  const bool native_available =
      avifCodecChoiceFromName("native") == AVIF_CODEC_CHOICE_NATIVE;
  const avifBool expected_native_decodable =
      (param.native_decodable && native_available) ? AVIF_TRUE : AVIF_FALSE;
  auto decoder = CreateDecoder(param.file_name);
  ASSERT_NE(decoder, nullptr);
  decoder->strictFlags = AVIF_STRICT_DISABLED;
  decoder->requestedSource = param.source;
  decoder->allowProgressive = param.allow_progressive;
  decoder->imageContentToDecode = AVIF_IMAGE_CONTENT_ALL;
  EXPECT_EQ(decoder->nativeDecodable, AVIF_FALSE);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->nativeDecodable, expected_native_decodable);
  if (!expected_native_decodable) {
    return;
  }
  decoder->codecChoice = AVIF_CODEC_CHOICE_NATIVE;
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  // The value is kept after decoding.
  EXPECT_EQ(decoder->nativeDecodable, expected_native_decodable);
}

INSTANTIATE_TEST_SUITE_P(
    NativeDecodableTestInstance, NativeDecodableTest,
    testing::ValuesIn(std::vector<NativeDecodableParam>{
        {"white_1x1.avif", AVIF_DECODER_SOURCE_AUTO, AVIF_FALSE, true},
        {"alpha.avif", AVIF_DECODER_SOURCE_AUTO, AVIF_FALSE, true},
        {"sofa_grid1x5_420.avif", AVIF_DECODER_SOURCE_AUTO, AVIF_FALSE, true},
        {"overlay_yellow_bg.avif", AVIF_DECODER_SOURCE_AUTO, AVIF_FALSE, true},
        {"seine_sdr_gainmap_srgb.avif", AVIF_DECODER_SOURCE_AUTO, AVIF_FALSE,
         true},
        {"colors-animated-8bpc.avif", AVIF_DECODER_SOURCE_AUTO, AVIF_FALSE,
         false},
        {"colors-animated-8bpc.avif", AVIF_DECODER_SOURCE_PRIMARY_ITEM,
         AVIF_FALSE, true},
        {"progressive/progressive_quality_change.avif",
         AVIF_DECODER_SOURCE_AUTO, AVIF_FALSE, false},
        {"progressive/progressive_quality_change.avif",
         AVIF_DECODER_SOURCE_AUTO, AVIF_TRUE, false},
        {"progressive/tiger_3layer_1res.avif", AVIF_DECODER_SOURCE_AUTO,
         AVIF_FALSE, false},
    }));

INSTANTIATE_TEST_SUITE_P(
    Codecs, DecoderTest, testing::ValuesIn(GetCodecChoices()),
    [](const testing::TestParamInfo<avifCodecChoice>& info) {
      return GetCodecName(info.param);
    });

TEST_P(DecoderTest, DecodeWithOptionsNullAndDisabled) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto ref_decoder = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(ref_decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(ref_decoder.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNthImage(ref_decoder.get(), 0), AVIF_RESULT_OK);

  auto decoder = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNextImageWithOptions(decoder.get(), nullptr),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *ref_decoder->image));
  ASSERT_EQ(avifDecoderNthImageWithOptions(decoder.get(), 0, nullptr),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *ref_decoder->image));

  // Zero-initialized options are the default options.
  const avifDecodeOptions default_opts{};
  ASSERT_EQ(avifDecoderReset(decoder.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNextImageWithOptions(decoder.get(), &default_opts),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *ref_decoder->image));
  ASSERT_EQ(avifDecoderNthImageWithOptions(decoder.get(), 0, &default_opts),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *ref_decoder->image));

  // When enableCropRect is AVIF_FALSE, cropRect values must be ignored.
  avifDecodeOptions disabled_opts{
      .enableCropRect = AVIF_FALSE,
      .cropRect = {.x = 99999, .y = 99999, .width = 0, .height = 0},
  };
  ASSERT_EQ(avifDecoderReset(decoder.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNthImageWithOptions(decoder.get(), 0, &disabled_opts),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *ref_decoder->image));
}

TEST_P(DecoderTest, DecodeRegionGridAndRgbParity) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  for (const char* filename :
       {"sofa_grid1x5_420.avif", "color_grid_alpha_nogrid.avif",
        "paris_10bpc.avif"}) {
    SCOPED_TRACE(filename);
    auto ref_decoder = CreateDecoder(filename);
    ASSERT_NE(ref_decoder, nullptr);
    ASSERT_EQ(avifDecoderParse(ref_decoder.get()), AVIF_RESULT_OK);
    ASSERT_EQ(avifDecoderNthImage(ref_decoder.get(), 0), AVIF_RESULT_OK);

    const uint32_t w = ref_decoder->image->width;
    const uint32_t h = ref_decoder->image->height;
    // Odd x, y, width, height; x and y round down to even only along
    // chroma-subsampled axes.
    const avifDecodeOptions opts{
        .enableCropRect = AVIF_TRUE,
        .cropRect = {.x = (w / 4) | 1u,
                     .y = (h / 4) | 1u,
                     .width = (w / 2) | 1u,
                     .height = (h / 2) | 1u},
    };
    avifPixelFormatInfo format_info{};
    avifGetPixelFormatInfo(ref_decoder->image->yuvFormat, &format_info);
    const avifCropRect aligned_rect{
        .x = (!format_info.monochrome && format_info.chromaShiftX == 1)
                 ? (opts.cropRect.x & ~1u)
                 : opts.cropRect.x,
        .y = (!format_info.monochrome && format_info.chromaShiftY == 1)
                 ? (opts.cropRect.y & ~1u)
                 : opts.cropRect.y,
        .width = opts.cropRect.width,
        .height = opts.cropRect.height,
    };

    ImagePtr view(avifImageCreateEmpty());
    ASSERT_NE(view, nullptr);
    ASSERT_EQ(
        avifImageSetViewRect(view.get(), ref_decoder->image, &aligned_rect),
        AVIF_RESULT_OK);

    auto decoder = CreateDecoder(filename);
    ASSERT_NE(decoder, nullptr);
    ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
    ASSERT_EQ(avifDecoderNthImageWithOptions(decoder.get(), 0, &opts),
              AVIF_RESULT_OK);
    EXPECT_TRUE(CompareImages(*decoder->image, *view));

    // Also verify YUV-to-RGB conversion on the region-decoded image matches the
    // view byte-for-byte.
    AvifRgbImage rgb_region(decoder->image, decoder->image->depth,
                            AVIF_RGB_FORMAT_RGBA);
    AvifRgbImage rgb_view(view.get(), view->depth, AVIF_RGB_FORMAT_RGBA);
    ASSERT_EQ(avifImageYUVToRGB(decoder->image, &rgb_region), AVIF_RESULT_OK);
    ASSERT_EQ(avifImageYUVToRGB(view.get(), &rgb_view), AVIF_RESULT_OK);
    const size_t row_bytes =
        rgb_region.width * avifRGBImagePixelSize(&rgb_region);
    for (uint32_t y = 0; y < rgb_region.height; ++y) {
      EXPECT_TRUE(testutil::AreByteSequencesEqual(
          rgb_region.pixels + y * rgb_region.rowBytes, row_bytes,
          rgb_view.pixels + y * rgb_view.rowBytes, row_bytes))
          << "row " << y;
    }
  }
}

TEST_P(DecoderTest, DecodeRegionGainMapAnimatedAndErrors) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  // 1. Gainmap-only region decode succeeds; Color + GainMap returns
  // AVIF_RESULT_NOT_IMPLEMENTED.
  auto gm_ref = CreateDecoder("seine_sdr_gainmap_srgb.avif");
  ASSERT_NE(gm_ref, nullptr);
  gm_ref->imageContentToDecode = AVIF_IMAGE_CONTENT_GAIN_MAP;
  ASSERT_EQ(avifDecoderParse(gm_ref.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNthImage(gm_ref.get(), 0), AVIF_RESULT_OK);

  const avifDecodeOptions gm_opts{
      .enableCropRect = AVIF_TRUE,
      .cropRect = {.x = 10, .y = 20, .width = 80, .height = 60},
  };
  auto gm_dec = CreateDecoder("seine_sdr_gainmap_srgb.avif");
  ASSERT_NE(gm_dec, nullptr);
  gm_dec->imageContentToDecode = AVIF_IMAGE_CONTENT_GAIN_MAP;
  ASSERT_EQ(avifDecoderParse(gm_dec.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNthImageWithOptions(gm_dec.get(), 0, &gm_opts),
            AVIF_RESULT_OK);
  // Only the gain map is decoded and cropped. The color image keeps its
  // uncropped dimensions and has no pixels.
  EXPECT_EQ(gm_dec->image->width, gm_ref->image->width);
  EXPECT_EQ(gm_dec->image->height, gm_ref->image->height);
  EXPECT_EQ(gm_dec->image->yuvPlanes[AVIF_CHAN_Y], nullptr);
  ASSERT_NE(gm_dec->image->gainMap, nullptr);
  ASSERT_NE(gm_dec->image->gainMap->image, nullptr);
  ImagePtr gm_view(avifImageCreateEmpty());
  ASSERT_EQ(avifImageSetViewRect(gm_view.get(), gm_ref->image->gainMap->image,
                                 &gm_opts.cropRect),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*gm_dec->image->gainMap->image, *gm_view));

  auto gm_both = CreateDecoder("seine_sdr_gainmap_srgb.avif");
  ASSERT_NE(gm_both, nullptr);
  gm_both->imageContentToDecode = AVIF_IMAGE_CONTENT_ALL;
  ASSERT_EQ(avifDecoderParse(gm_both.get()), AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderNthImageWithOptions(gm_both.get(), 0, &gm_opts),
            AVIF_RESULT_NOT_IMPLEMENTED);

  // Gainmap-only region decode of an image without a gain map fails.
  auto no_gm = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(no_gm, nullptr);
  no_gm->imageContentToDecode = AVIF_IMAGE_CONTENT_GAIN_MAP;
  ASSERT_EQ(avifDecoderParse(no_gm.get()), AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderNthImageWithOptions(no_gm.get(), 0, &gm_opts),
            AVIF_RESULT_NO_CONTENT);

  // 2. Animated sequence with NextImageWithOptions and NthImageWithOptions.
  // The native AV1 decoder does not support inter frames.
  if (codec_choice() != AVIF_CODEC_CHOICE_NATIVE) {
    const avifDecodeOptions anim_opts{
        .enableCropRect = AVIF_TRUE,
        .cropRect = {.x = 4, .y = 6, .width = 20, .height = 20},
    };
    auto anim_ref = CreateDecoder("colors-animated-8bpc.avif");
    ASSERT_NE(anim_ref, nullptr);
    ASSERT_EQ(avifDecoderParse(anim_ref.get()), AVIF_RESULT_OK);
    auto anim_dec = CreateDecoder("colors-animated-8bpc.avif");
    ASSERT_NE(anim_dec, nullptr);
    ASSERT_EQ(avifDecoderParse(anim_dec.get()), AVIF_RESULT_OK);
    for (int i = 0; i < anim_ref->imageCount; ++i) {
      SCOPED_TRACE(i);
      ASSERT_EQ(avifDecoderNextImage(anim_ref.get()), AVIF_RESULT_OK);
      ASSERT_EQ(avifDecoderNextImageWithOptions(anim_dec.get(), &anim_opts),
                AVIF_RESULT_OK);
      ImagePtr anim_view(avifImageCreateEmpty());
      ASSERT_EQ(avifImageSetViewRect(anim_view.get(), anim_ref->image,
                                     &anim_opts.cropRect),
                AVIF_RESULT_OK);
      EXPECT_TRUE(CompareImages(*anim_dec->image, *anim_view));
    }
    ASSERT_EQ(avifDecoderNthImage(anim_ref.get(), 3), AVIF_RESULT_OK);
    ASSERT_EQ(avifDecoderNthImageWithOptions(anim_dec.get(), 3, &anim_opts),
              AVIF_RESULT_OK);
    ImagePtr anim_view(avifImageCreateEmpty());
    ASSERT_EQ(avifImageSetViewRect(anim_view.get(), anim_ref->image,
                                   &anim_opts.cropRect),
              AVIF_RESULT_OK);
    EXPECT_TRUE(CompareImages(*anim_dec->image, *anim_view));
  }

  // 3. Progressive or incremental + cropRect returns
  // AVIF_RESULT_NOT_IMPLEMENTED (region decoding requires the whole file).
  auto inc_dec = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(inc_dec, nullptr);
  inc_dec->allowIncremental = AVIF_TRUE;
  ASSERT_EQ(avifDecoderParse(inc_dec.get()), AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderNextImageWithOptions(inc_dec.get(), &gm_opts),
            AVIF_RESULT_NOT_IMPLEMENTED);
  auto prog_dec = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(prog_dec, nullptr);
  prog_dec->allowProgressive = AVIF_TRUE;
  ASSERT_EQ(avifDecoderParse(prog_dec.get()), AVIF_RESULT_OK);
  EXPECT_EQ(avifDecoderNextImageWithOptions(prog_dec.get(), &gm_opts),
            AVIF_RESULT_NOT_IMPLEMENTED);

  // 4. Zero-width and out-of-bounds crops return AVIF_RESULT_INVALID_ARGUMENT.
  auto err_dec = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(err_dec, nullptr);
  ASSERT_EQ(avifDecoderParse(err_dec.get()), AVIF_RESULT_OK);
  const avifDecodeOptions zero_opts{
      .enableCropRect = AVIF_TRUE,
      .cropRect = {.x = 0, .y = 0, .width = 0, .height = 10},
  };
  EXPECT_EQ(avifDecoderNthImageWithOptions(err_dec.get(), 0, &zero_opts),
            AVIF_RESULT_INVALID_ARGUMENT);
  const avifDecodeOptions out_of_bounds_opts{
      .enableCropRect = AVIF_TRUE,
      .cropRect = {.x = err_dec->image->width - 5,
                   .y = 0,
                   .width = 6,
                   .height = 10},
  };
  EXPECT_EQ(
      avifDecoderNthImageWithOptions(err_dec.get(), 0, &out_of_bounds_opts),
      AVIF_RESULT_INVALID_ARGUMENT);
  EXPECT_EQ(avifDecoderNextImageWithOptions(err_dec.get(), &out_of_bounds_opts),
            AVIF_RESULT_INVALID_ARGUMENT);
}

// A NextImage call past the last image after a region decode fails without
// modifying the decoded image, which must stay consistent with its planes.
TEST_P(DecoderTest, DecodeRegionThenNoImagesRemaining) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto ref_decoder = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(ref_decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(ref_decoder.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNthImage(ref_decoder.get(), 0), AVIF_RESULT_OK);
  const avifDecodeOptions opts{
      .enableCropRect = AVIF_TRUE,
      .cropRect = {.x = 100, .y = 200, .width = 300, .height = 250},
  };
  ImagePtr view(avifImageCreateEmpty());
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(
      avifImageSetViewRect(view.get(), ref_decoder->image, &opts.cropRect),
      AVIF_RESULT_OK);

  auto decoder = CreateDecoder("sofa_grid1x5_420.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNextImageWithOptions(decoder.get(), &opts),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *view));
  EXPECT_EQ(avifDecoderNextImage(decoder.get()),
            AVIF_RESULT_NO_IMAGES_REMAINING);
  EXPECT_EQ(decoder->image->width, opts.cropRect.width);
  EXPECT_EQ(decoder->image->height, opts.cropRect.height);
  EXPECT_TRUE(CompareImages(*decoder->image, *view));
  EXPECT_EQ(avifDecoderNextImageWithOptions(decoder.get(), &opts),
            AVIF_RESULT_NO_IMAGES_REMAINING);
  EXPECT_TRUE(CompareImages(*decoder->image, *view));
}

// Region decodes interleaved with full decodes on the same decoder. The full
// decodes leave planes pointing into codec-owned memory, which a region decode
// must neither read nor expose.
TEST_P(DecoderTest, DecodeRegionInterleavedWithFullDecodes) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  for (const char* filename : {"alpha.avif", "color_grid_alpha_nogrid.avif",
                               "sofa_grid1x5_420.avif"}) {
    SCOPED_TRACE(filename);
    auto ref_decoder = CreateDecoder(filename);
    ASSERT_NE(ref_decoder, nullptr);
    ASSERT_EQ(avifDecoderParse(ref_decoder.get()), AVIF_RESULT_OK);
    ASSERT_EQ(avifDecoderNthImage(ref_decoder.get(), 0), AVIF_RESULT_OK);
    const uint32_t w = ref_decoder->image->width;
    const uint32_t h = ref_decoder->image->height;
    const avifDecodeOptions opts{
        .enableCropRect = AVIF_TRUE,
        .cropRect = {.x = (w / 4) & ~1u,
                     .y = (h / 4) & ~1u,
                     .width = w / 2,
                     .height = h / 2},
    };
    ImagePtr view(avifImageCreateEmpty());
    ASSERT_NE(view, nullptr);
    ASSERT_EQ(
        avifImageSetViewRect(view.get(), ref_decoder->image, &opts.cropRect),
        AVIF_RESULT_OK);

    auto decoder = CreateDecoder(filename);
    ASSERT_NE(decoder, nullptr);
    ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
    for (int i = 0; i < 2; ++i) {
      ASSERT_EQ(avifDecoderNthImage(decoder.get(), 0), AVIF_RESULT_OK);
      EXPECT_TRUE(CompareImages(*decoder->image, *ref_decoder->image));
      ASSERT_EQ(avifDecoderNthImageWithOptions(decoder.get(), 0, &opts),
                AVIF_RESULT_OK);
      EXPECT_TRUE(CompareImages(*decoder->image, *view));
    }
  }
}

// Region decoding of a 120x160 image with a clean aperture (crop rectangle
// (12, 14, 96, 132)), rotation and mirroring. The region is relative to the
// clean aperture, which is not applied to the decoded image.
TEST_P(DecoderTest, DecodeRegionCleanAperture) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const char* filename = "region/region_clap_irot_imir.avif";
  auto ref_decoder = CreateDecoder(filename);
  ASSERT_NE(ref_decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(ref_decoder.get()), AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderNthImage(ref_decoder.get(), 0), AVIF_RESULT_OK);
  ASSERT_EQ(ref_decoder->image->width, 120u);
  ASSERT_EQ(ref_decoder->image->height, 160u);
  ASSERT_EQ(ref_decoder->image->transformFlags,
            AVIF_TRANSFORM_CLAP | AVIF_TRANSFORM_IROT | AVIF_TRANSFORM_IMIR);

  auto decoder = CreateDecoder(filename);
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);

  // Odd user crop (5, 7, 41, 39) inside clap (12, 14, 96, 132) -> (17, 21),
  // aligned down to (16, 20).
  const avifDecodeOptions opts{
      .enableCropRect = AVIF_TRUE,
      .cropRect = {.x = 5, .y = 7, .width = 41, .height = 39},
  };
  ASSERT_EQ(avifDecoderNextImageWithOptions(decoder.get(), &opts),
            AVIF_RESULT_OK);
  EXPECT_EQ(decoder->image->width, 41u);
  EXPECT_EQ(decoder->image->height, 39u);
  // CLAP flag must be cleared on cropped output while IROT/IMIR remain intact.
  EXPECT_EQ(decoder->image->transformFlags,
            AVIF_TRANSFORM_IROT | AVIF_TRANSFORM_IMIR);

  const avifCropRect expected_rect{.x = 16, .y = 20, .width = 41, .height = 39};
  ImagePtr expected_view(avifImageCreateEmpty());
  ASSERT_NE(expected_view, nullptr);
  ASSERT_EQ(avifImageSetViewRect(expected_view.get(), ref_decoder->image,
                                 &expected_rect),
            AVIF_RESULT_OK);
  EXPECT_TRUE(CompareImages(*decoder->image, *expected_view));

  // avifDecoderNthImageProperties still reports uncropped dimensions and CLAP.
  ImagePtr props(avifImageCreateEmpty());
  ASSERT_NE(props, nullptr);
  ASSERT_EQ(avifDecoderNthImageProperties(decoder.get(), 0, props.get()),
            AVIF_RESULT_OK);
  EXPECT_EQ(props->width, 120u);
  EXPECT_EQ(props->height, 160u);
  EXPECT_NE(props->transformFlags & AVIF_TRANSFORM_CLAP, 0u);

  // Crop exceeding clap bounds (90 + 10 > 96, even though 12 + 100 <= 120) must
  // fail with AVIF_RESULT_INVALID_ARGUMENT.
  const avifDecodeOptions out_of_clap{
      .enableCropRect = AVIF_TRUE,
      .cropRect = {.x = 90, .y = 0, .width = 10, .height = 10},
  };
  EXPECT_EQ(avifDecoderNthImageWithOptions(decoder.get(), 0, &out_of_clap),
            AVIF_RESULT_INVALID_ARGUMENT);

  // Subsequent uncropped decode restores original dimensions and CLAP flag.
  ASSERT_EQ(avifDecoderNthImage(decoder.get(), 0), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->image->width, 120u);
  EXPECT_EQ(decoder->image->height, 160u);
  EXPECT_NE(decoder->image->transformFlags & AVIF_TRANSFORM_CLAP, 0u);
  EXPECT_TRUE(CompareImages(*decoder->image, *ref_decoder->image));
}

}  // namespace
}  // namespace avif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc != 2) {
    std::cerr << "There must be exactly one argument containing the path to "
                 "the test data folder"
              << std::endl;
    return 1;
  }
  avif::data_path = argv[1];
  return RUN_ALL_TESTS();
}
