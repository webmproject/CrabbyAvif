// Copyright 2023 Google LLC
// SPDX-License-Identifier: BSD-2-Clause

#include <vector>

#include "avif/avif.h"
#include "aviftest_helpers.h"
#include "gtest/gtest.h"

namespace avif {
namespace {

constexpr uint8_t kWidth = 4;
constexpr uint8_t kHeight = 4;
constexpr uint8_t kPlaneSize = 16;
constexpr uint8_t kUOffset = 16;
constexpr uint8_t kVOffset = 32;
constexpr uint8_t kWhite[] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
constexpr uint8_t kWhiteRGBA[] = {0xff, 0xff, 0xff, 0xff};

TEST(AvifDecodeTest, YUVToRGBConversion) {
  ImagePtr image(avifImageCreate(kWidth, kHeight, 8, AVIF_PIXEL_FORMAT_YUV444));
  ASSERT_NE(image, nullptr);
  fprintf(stderr, "### im here\n");
  ASSERT_EQ(avifImageAllocatePlanes(image.get(), AVIF_PLANES_YUV),
            AVIF_RESULT_OK);
  memcpy(image->yuvPlanes[0], kWhite, kPlaneSize);
  memcpy(image->yuvPlanes[1], kWhite + kUOffset, kPlaneSize);
  memcpy(image->yuvPlanes[2], kWhite + kVOffset, kPlaneSize);
  avifRGBImage rgb;
  avifRGBImageSetDefaults(&rgb, image.get());
  std::vector<uint8_t> rgb_pixels(kPlaneSize * 4);
  rgb.pixels = rgb_pixels.data();
  rgb.rowBytes = kWidth * 4;
  ASSERT_EQ(avifImageYUVToRGB(image.get(), &rgb), AVIF_RESULT_OK);
  for (int i = 0; i < kPlaneSize; ++i) {
    fprintf(stderr, "### %d %d %d %d\n", rgb.pixels[i * 4],
            rgb.pixels[i * 4 + 1], rgb.pixels[i * 4 + 2],
            rgb.pixels[i * 4 + 3]);
    EXPECT_EQ(rgb.pixels[i * 4], kWhiteRGBA[0]);
    EXPECT_EQ(rgb.pixels[i * 4 + 1], kWhiteRGBA[1]);
    EXPECT_EQ(rgb.pixels[i * 4 + 2], kWhiteRGBA[2]);
    EXPECT_EQ(rgb.pixels[i * 4 + 3], kWhiteRGBA[3]);
  }
  avifImageFreePlanes(image.get(), AVIF_PLANES_YUV);
}

}  // namespace
}  // namespace avif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
