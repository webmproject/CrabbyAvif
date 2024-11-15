// Copyright 2024 Google LLC
// SPDX-License-Identifier: BSD-2-Clause

#include <cstring>
#include <iostream>
#include <string>

#include "avif/avif.h"
#include "aviftest_helpers.h"
#include "gtest/gtest.h"

namespace avif {
namespace {

// Used to pass the data folder path to the GoogleTest suites.
const char* data_path = nullptr;

class ImageTest : public testing::TestWithParam<const char*> {};

TEST_P(ImageTest, ImageCopy) {
  if (!testutil::Av1DecoderAvailable()) {
    GTEST_SKIP() << "AV1 Codec unavailable, skip test.";
  }
  const char* file_name = GetParam();
  DecoderPtr decoder(avifDecoderCreate());
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(avifDecoderSetIOFile(decoder.get(),
                                 (std::string(data_path) + file_name).c_str()),
            AVIF_RESULT_OK);
  ASSERT_EQ(avifDecoderParse(decoder.get()), AVIF_RESULT_OK);
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

INSTANTIATE_TEST_SUITE_P(Some, ImageTest,
                         testing::ValuesIn({"paris_10bpc.avif", "alpha.avif",
                                            "colors-animated-8bpc.avif"}));

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
