// Copyright 2025 Google LLC
// SPDX-License-Identifier: BSD-2-Clause

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

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

TEST(DecoderTest, AlphaNoIspe) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  // See https://github.com/AOMediaCodec/libavif/pull/745.
  auto decoder = CreateDecoder("alpha_noispe.avif");
  ASSERT_NE(decoder, nullptr);
  // By default, loose files are refused. Cast to avoid C4389 Windows warning.
  EXPECT_EQ(decoder->strictFlags, (avifStrictFlags)AVIF_STRICT_ENABLED);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_BMFF_PARSE_FAILED);
  // Allow this kind of file specifically.
  decoder->strictFlags = (avifStrictFlags)AVIF_STRICT_ENABLED &
                         ~(avifStrictFlags)AVIF_STRICT_ALPHA_ISPE_REQUIRED;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->alphaPresent, AVIF_TRUE);
  EXPECT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_OK);
  EXPECT_NE(decoder->image->alphaPlane, nullptr);
  EXPECT_GT(decoder->image->alphaRowBytes, 0u);
}

TEST(DecoderTest, AnimatedImage) {
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

TEST(DecoderTest, AnimatedImageWithSourceSetToPrimaryItem) {
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

TEST(DecoderTest, OneShotDecodeFile) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const char* file_name = "sofa_grid1x5_420.avif";
  DecoderPtr decoder(avifDecoderCreate());
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

TEST(DecoderTest, OneShotDecodeMemory) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const char* file_name = "sofa_grid1x5_420.avif";
  auto file_data = testutil::read_file(GetFilename(file_name).c_str());
  DecoderPtr decoder(avifDecoderCreate());
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
  avifROData* src = (avifROData*)io->data;
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

TEST(DecoderTest, OneShotDecodeCustomIO) {
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
               .data = static_cast<void*>(&ro_data)};
  DecoderPtr decoder(avifDecoderCreate());
  ASSERT_NE(decoder, nullptr);
  avifDecoderSetIO(decoder.get(), &io);
  avifImage image;
  ASSERT_EQ(avifDecoderRead(decoder.get(), &image), AVIF_RESULT_OK);
  EXPECT_EQ(image.width, 1024);
  EXPECT_EQ(image.height, 770);
  EXPECT_EQ(image.depth, 8);
}

TEST(DecoderTest, NthImage) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  auto decoder = CreateDecoder("colors-animated-8bpc.avif");
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
  EXPECT_EQ(decoder->compressionFormat, COMPRESSION_FORMAT_AVIF);
  EXPECT_EQ(decoder->imageCount, 5);
  EXPECT_EQ(avifDecoderNthImage(decoder.get(), 3), AVIF_RESULT_OK);
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

TEST(DecoderTest, ColorGridAlphaNoGrid) {
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

TEST(DecoderTest, GainMapGrid) {
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
  EXPECT_NE(decoder->image->gainMap, nullptr);
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

TEST(DecoderTest, GainMapOriented) {
  auto decoder = CreateDecoder(("gainmap_oriented.avif"));
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode |= AVIF_IMAGE_CONTENT_GAIN_MAP;
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);

  // Verify that the transformative properties were kept.
  EXPECT_EQ(decoder->image->transformFlags,
            AVIF_TRANSFORM_IROT | AVIF_TRANSFORM_IMIR);
  EXPECT_EQ(decoder->image->irot.angle, 1);
  EXPECT_EQ(decoder->image->imir.axis, 0);
  EXPECT_EQ(decoder->image->gainMap->image->transformFlags,
            AVIF_TRANSFORM_NONE);
}

TEST(DecoderTest, IgnoreGainMapButReadMetadata) {
  auto decoder = CreateDecoder(("seine_sdr_gainmap_srgb.avif"));
  ASSERT_NE(decoder, nullptr);
  auto result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
  avifImage* decoded = decoder->image;
  ASSERT_NE(decoded, nullptr);

  // Verify that the gain map was detected...
  EXPECT_NE(decoder->image->gainMap, nullptr);
  // ... but not decoded because enableDecodingGainMap is false by default.
  EXPECT_EQ(decoded->gainMap->image, nullptr);
  // Check that the gain map metadata WAS populated.
  EXPECT_EQ(decoded->gainMap->alternateHdrHeadroom.n, 13);
  EXPECT_EQ(decoded->gainMap->alternateHdrHeadroom.d, 10);
}

TEST(DecoderTest, IgnoreColorAndAlpha) {
  auto decoder = CreateDecoder(("seine_sdr_gainmap_srgb.avif"));
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
  EXPECT_NE(decoder->image->gainMap, nullptr);
  ASSERT_NE(decoded->gainMap->image, nullptr);
  // Including pixels.
  EXPECT_GT(decoded->gainMap->image->yuvRowBytes[0], 0u);
}

TEST(DecoderTest, IgnoreAll) {
  auto decoder = CreateDecoder(("seine_sdr_gainmap_srgb.avif"));
  ASSERT_NE(decoder, nullptr);
  decoder->imageContentToDecode = AVIF_IMAGE_CONTENT_NONE;
  auto result = avifDecoderParse(decoder.get());
  ASSERT_EQ(result, AVIF_RESULT_OK)
      << avifResultToString(result) << " " << decoder->diag.error;
  avifImage* decoded = decoder->image;
  ASSERT_NE(decoded, nullptr);

  EXPECT_NE(decoder->image->gainMap, nullptr);
  ASSERT_EQ(decoder->image->gainMap->image, nullptr);

  // But trying to access the next image should give an error because both
  // ignoreColorAndAlpha and enableDecodingGainMap are set.
  ASSERT_EQ(avifDecoderNextImage(decoder.get()), AVIF_RESULT_NO_CONTENT);
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

TEST(DecoderTest, Progressive) {
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
    DecoderPtr decoder(avifDecoderCreate());
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
  auto decoder = CreateDecoder(("paris_icc_exif_xmp.avif"));
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

class ImageTest : public testing::TestWithParam<const char*> {};

TEST_P(ImageTest, ImageCopy) {
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
  EXPECT_EQ(decoder->image->width, image2->width);
  EXPECT_EQ(decoder->image->height, image2->height);
  EXPECT_EQ(decoder->image->depth, image2->depth);
  EXPECT_EQ(decoder->image->yuvFormat, image2->yuvFormat);
  EXPECT_EQ(decoder->image->yuvRange, image2->yuvRange);
  for (int plane = 0; plane < 3; ++plane) {
    EXPECT_EQ(decoder->image->yuvPlanes[plane] == nullptr,
              image2->yuvPlanes[plane] == nullptr);
    if (decoder->image->yuvPlanes[plane] == nullptr) continue;
    EXPECT_EQ(decoder->image->yuvRowBytes[plane], image2->yuvRowBytes[plane]);
    EXPECT_NE(decoder->image->yuvPlanes[plane], image2->yuvPlanes[plane]);
    const auto plane_height = avifImagePlaneHeight(decoder->image, plane);
    const auto plane_size = plane_height * decoder->image->yuvRowBytes[plane];
    EXPECT_EQ(memcmp(decoder->image->yuvPlanes[plane], image2->yuvPlanes[plane],
                     plane_size),
              0);
  }
  EXPECT_EQ(decoder->image->alphaPlane == nullptr,
            image2->alphaPlane == nullptr);
  if (decoder->image->alphaPlane != nullptr) {
    EXPECT_EQ(decoder->image->alphaRowBytes, image2->alphaRowBytes);
    EXPECT_NE(decoder->image->alphaPlane, image2->alphaPlane);
    const auto plane_size =
        decoder->image->height * decoder->image->alphaRowBytes;
    EXPECT_EQ(
        memcmp(decoder->image->alphaPlane, image2->alphaPlane, plane_size), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(ImageTestInstance, ImageTest,
                         testing::ValuesIn({"paris_10bpc.avif", "alpha.avif",
                                            "colors-animated-8bpc.avif"}));

TEST(DecoderTest, SetRawIO) {
  DecoderPtr decoder(avifDecoderCreate());
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
}

TEST(DecoderTest, SetCustomIO) {
  DecoderPtr decoder(avifDecoderCreate());
  ASSERT_NE(decoder, nullptr);
  auto data =
      testutil::read_file(GetFilename("colors-animated-8bpc.avif").c_str());
  avifROData ro_data = {.data = data.data(), .size = data.size()};
  avifIO io = {.destroy = nullptr,
               .read = io_read,
               .sizeHint = data.size(),
               .persistent = false,
               .data = static_cast<void*>(&ro_data)};
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