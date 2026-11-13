/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include "avif/avif.h"
#include "gtest/gtest.h"
#include "testutil.h"

namespace avif {
namespace {

#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER) || defined(HWADDRESS_SANITIZER)
#define CRABBYAVIF_SANITIZER_BUILD
#endif

// Used to pass the data folder path to the GoogleTest suites.
const char* data_path = nullptr;

TEST(ImageTest, Create) {
  ImagePtr image(avifImageCreateEmpty());
  EXPECT_NE(image, nullptr);
  image.reset(avifImageCreate(0, 0, 0, AVIF_PIXEL_FORMAT_NONE));
  EXPECT_NE(image, nullptr);
  image.reset(avifImageCreate(1, 1, /*depth=*/1, AVIF_PIXEL_FORMAT_NONE));
  EXPECT_NE(image, nullptr);
  image.reset(avifImageCreate(64, 64, /*depth=*/8, AVIF_PIXEL_FORMAT_NONE));
  EXPECT_NE(image, nullptr);
  image.reset(avifImageCreate(std::numeric_limits<uint32_t>::max(),
                              std::numeric_limits<uint32_t>::max(),
                              /*depth=*/16, AVIF_PIXEL_FORMAT_NONE));
  EXPECT_NE(image, nullptr);
}

void TestAllocation(uint32_t width, uint32_t height, uint32_t depth,
                    avifPixelFormat yuv_format, avifPlanesFlags planes,
                    bool expect_success) {
  ImagePtr image(avifImageCreateEmpty());
  ASSERT_NE(image, nullptr);
  image->width = width;
  image->height = height;
  image->depth = depth;
  image->yuvFormat = yuv_format;
  auto res = avifImageAllocatePlanes(image.get(), planes);
  if (expect_success) {
    ASSERT_EQ(res, AVIF_RESULT_OK);
    if (yuv_format != AVIF_PIXEL_FORMAT_NONE && (planes & AVIF_PLANES_YUV)) {
      EXPECT_NE(image->yuvPlanes[AVIF_CHAN_Y], nullptr);
      if (yuv_format != AVIF_PIXEL_FORMAT_YUV400) {
        EXPECT_NE(image->yuvPlanes[AVIF_CHAN_U], nullptr);
        EXPECT_NE(image->yuvPlanes[AVIF_CHAN_V], nullptr);
      } else {
        EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_U], nullptr);
        EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_V], nullptr);
      }
    } else {
      EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_Y], nullptr);
      EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_U], nullptr);
      EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_V], nullptr);
    }
    if (planes & AVIF_PLANES_A) {
      EXPECT_NE(image->alphaPlane, nullptr);
    } else {
      EXPECT_EQ(image->alphaPlane, nullptr);
    }
  } else {
    ASSERT_NE(res, AVIF_RESULT_OK);
    EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_Y], nullptr);
    EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_U], nullptr);
    EXPECT_EQ(image->yuvPlanes[AVIF_CHAN_V], nullptr);
    EXPECT_EQ(image->alphaPlane, nullptr);
  }
}

class ImageAllocationTest
    : public testing::TestWithParam<
          std::tuple<avifPixelFormat, avifPlanesFlag, /*depth=*/int>> {};

TEST_P(ImageAllocationTest, VariousCases) {
  const auto& param = GetParam();
  const auto yuv_format = std::get<0>(param);
  const auto planes = std::get<1>(param);
  const auto depth = std::get<2>(param);
  // Minimum valid image dimensions.
  TestAllocation(1, 1, depth, yuv_format, planes, true);
#if !defined(CRABBYAVIF_SANITIZER_BUILD)
  // Maximum valid image dimensions. This allocation is too large for
  // sanitizers.
  TestAllocation(CRABBY_AVIF_DEFAULT_IMAGE_DIMENSION_LIMIT,
                 CRABBY_AVIF_DEFAULT_IMAGE_DIMENSION_LIMIT, depth, yuv_format,
                 planes, true);
#endif
  // Invalid (too large).
  TestAllocation((1 << 30), 1, depth, yuv_format, planes, false);
}

INSTANTIATE_TEST_SUITE_P(
    All, ImageAllocationTest,
    testing::Combine(
        testing::Values(AVIF_PIXEL_FORMAT_NONE, AVIF_PIXEL_FORMAT_YUV444,
                        AVIF_PIXEL_FORMAT_YUV422, AVIF_PIXEL_FORMAT_YUV420,
                        AVIF_PIXEL_FORMAT_YUV400),
        testing::Values(AVIF_PLANES_YUV, AVIF_PLANES_A, AVIF_PLANES_ALL),
        testing::Values(8, 10, 12)));

void TestEncoding(uint32_t width, uint32_t height, uint32_t depth,
                  avifResult expected_result) {
  ImagePtr image(avifImageCreateEmpty());
  ASSERT_NE(image, nullptr);
  image->width = width;
  image->height = height;
  image->depth = depth;
  image->yuvFormat = AVIF_PIXEL_FORMAT_YUV444;

  // This is a fairly high number of bytes that can safely be allocated in this
  // test. The goal is to have something to give to libavif but libavif should
  // return an error before attempting to read all of it, so it does not matter
  // if there are fewer bytes than the provided image dimensions.
  static constexpr uint64_t kMaxAlloc = 1073741824;
  uint32_t row_bytes;
  size_t num_allocated_bytes;
  if (static_cast<uint64_t>(image->width) * image->height >
      kMaxAlloc / (avifImageUsesU16(image.get()) ? 2 : 1)) {
    row_bytes = 1024;  // Does not matter much.
    num_allocated_bytes = kMaxAlloc;
  } else {
    row_bytes = image->width * (avifImageUsesU16(image.get()) ? 2 : 1);
    num_allocated_bytes = row_bytes * image->height;
  }

  // Initialize pixels as 16b values to make sure values are valid for 10
  // and 12-bit depths. The array will be cast to uint8_t for 8-bit depth.
  std::vector<uint16_t> pixels(
      std::max(1lu, num_allocated_bytes / sizeof(uint16_t)), 400);
  uint8_t* bytes = reinterpret_cast<uint8_t*>(pixels.data());
  // Avoid avifImageAllocatePlanes() to exercise the checks at encoding.
  image->imageOwnsYUVPlanes = AVIF_FALSE;
  image->imageOwnsAlphaPlane = AVIF_FALSE;
  image->yuvRowBytes[AVIF_CHAN_Y] = row_bytes;
  image->yuvPlanes[AVIF_CHAN_Y] = bytes;
  image->yuvRowBytes[AVIF_CHAN_U] = row_bytes;
  image->yuvPlanes[AVIF_CHAN_U] = bytes;
  image->yuvRowBytes[AVIF_CHAN_V] = row_bytes;
  image->yuvPlanes[AVIF_CHAN_V] = bytes;
  image->alphaRowBytes = row_bytes;
  image->alphaPlane = bytes;

  // Try to encode.
  EncoderPtr encoder(avifEncoderCreate());
  ASSERT_NE(encoder, nullptr);
  encoder->speed = 10;
  AvifRwData encoded_avif;
  ASSERT_EQ(avifEncoderWrite(encoder.get(), image.get(), &encoded_avif),
            expected_result);
}

TEST(EncodingTest, VariousCases) {
  TestEncoding(1, 1, 8, AVIF_RESULT_OK);
  TestEncoding(101, 102, 8, AVIF_RESULT_OK);
#if !defined(CRABBYAVIF_SANITIZER_BUILD)
  // This allocation is too large for sanitizers.
  TestEncoding(CRABBY_AVIF_DEFAULT_IMAGE_DIMENSION_LIMIT / 2,
               CRABBY_AVIF_DEFAULT_IMAGE_DIMENSION_LIMIT / 2, 8,
               AVIF_RESULT_OK);
#endif
}

TEST(ImageTest, MetadataFunctions) {
  // Fill data with some values.
  constexpr int kSize = 10;
  std::vector<uint8_t> data;
  data.reserve(kSize);
  for (int i = 0; i < kSize; ++i) {
    data.push_back(i * 2);
  }

  ImagePtr image(avifImageCreateEmpty());
  ASSERT_NE(image, nullptr);

  for (auto* func : {avifImageSetMetadataExif, avifImageSetProfileICC,
                     avifImageSetMetadataXMP}) {
    const auto& metadata =
        (func == avifImageSetMetadataExif)
            ? image->exif
            : ((func == avifImageSetProfileICC) ? image->icc : image->xmp);
    EXPECT_NE(func(nullptr, nullptr, 0), AVIF_RESULT_OK);
    EXPECT_EQ(func(image.get(), nullptr, 0), AVIF_RESULT_OK);
    EXPECT_EQ(metadata.data, nullptr);
    EXPECT_EQ(metadata.size, 0);
    EXPECT_EQ(func(image.get(), data.data(), 0), AVIF_RESULT_OK);
    EXPECT_EQ(metadata.data, nullptr);
    EXPECT_EQ(metadata.size, 0);
    EXPECT_EQ(func(image.get(), data.data(), kSize), AVIF_RESULT_OK);
    EXPECT_NE(metadata.data, nullptr);
    EXPECT_NE(metadata.data, data.data());
    ASSERT_EQ(metadata.size, kSize);
    EXPECT_EQ(memcmp(metadata.data, data.data(), kSize), 0);
  }
}

TEST(RWDataTest, Set) {
  AvifRwData rw_data;
  const uint8_t data[] = {1, 2, 3, 4, 5};
  ASSERT_EQ(avifRWDataSet(&rw_data, data, sizeof(data)), AVIF_RESULT_OK);
  ASSERT_EQ(rw_data.size, sizeof(data));
  EXPECT_NE(rw_data.data, data);
  EXPECT_EQ(memcmp(rw_data.data, data, sizeof(data)), 0);

  // Overwrite with smaller, larger and same sized data.
  for (const std::vector<uint8_t>& new_data :
       {std::vector<uint8_t>{6, 7}, std::vector<uint8_t>(100, 8),
        std::vector<uint8_t>(100, 9)}) {
    ASSERT_EQ(avifRWDataSet(&rw_data, new_data.data(), new_data.size()),
              AVIF_RESULT_OK);
    ASSERT_EQ(rw_data.size, new_data.size());
    EXPECT_EQ(memcmp(rw_data.data, new_data.data(), new_data.size()), 0);
  }

  avifRWDataFree(&rw_data);
  EXPECT_EQ(rw_data.data, nullptr);
  EXPECT_EQ(rw_data.size, 0u);
}

TEST(RWDataTest, SetFromOwnBuffer) {
  AvifRwData rw_data;
  std::vector<uint8_t> data(16);
  for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i);
  ASSERT_EQ(avifRWDataSet(&rw_data, data.data(), data.size()), AVIF_RESULT_OK);

  // The source points into the buffer that is being replaced.
  ASSERT_EQ(avifRWDataSet(&rw_data, rw_data.data + 4, 8), AVIF_RESULT_OK);
  ASSERT_EQ(rw_data.size, 8u);
  EXPECT_EQ(memcmp(rw_data.data, data.data() + 4, 8), 0);

  // Same size as the existing buffer.
  ASSERT_EQ(avifRWDataSet(&rw_data, rw_data.data, rw_data.size),
            AVIF_RESULT_OK);
  ASSERT_EQ(rw_data.size, 8u);
  EXPECT_EQ(memcmp(rw_data.data, data.data() + 4, 8), 0);
}

TEST(RWDataTest, SetEmpty) {
  AvifRwData rw_data;
  const uint8_t data[] = {1, 2, 3};
  ASSERT_EQ(avifRWDataSet(&rw_data, data, sizeof(data)), AVIF_RESULT_OK);
  ASSERT_EQ(avifRWDataSet(&rw_data, nullptr, 0), AVIF_RESULT_OK);
  EXPECT_EQ(rw_data.data, nullptr);
  EXPECT_EQ(rw_data.size, 0u);
  // The AvifRwData destructor frees rw_data again, which must be a no-op.
}

TEST(RWDataTest, Realloc) {
  AvifRwData rw_data;
  const uint8_t data[] = {1, 2, 3};
  ASSERT_EQ(avifRWDataSet(&rw_data, data, sizeof(data)), AVIF_RESULT_OK);
  ASSERT_EQ(avifRWDataRealloc(&rw_data, 6), AVIF_RESULT_OK);
  ASSERT_EQ(rw_data.size, 6u);
  const uint8_t expected_grown[] = {1, 2, 3, 0, 0, 0};
  EXPECT_EQ(memcmp(rw_data.data, expected_grown, sizeof(expected_grown)), 0);
  ASSERT_EQ(avifRWDataRealloc(&rw_data, 2), AVIF_RESULT_OK);
  ASSERT_EQ(rw_data.size, 2u);
  EXPECT_EQ(memcmp(rw_data.data, data, 2), 0);
  ASSERT_EQ(avifRWDataRealloc(&rw_data, 0), AVIF_RESULT_OK);
  EXPECT_EQ(rw_data.data, nullptr);
  EXPECT_EQ(rw_data.size, 0u);
}

TEST(ImageTest, CopyViewRectAtRightEdge) {
  for (uint32_t depth : {8, 10}) {
    SCOPED_TRACE(depth);
    ImagePtr src(avifImageCreate(4, 4, depth, AVIF_PIXEL_FORMAT_YUV444));
    ASSERT_NE(src, nullptr);
    ASSERT_EQ(avifImageAllocatePlanes(src.get(), AVIF_PLANES_ALL),
              AVIF_RESULT_OK);
    const size_t pixel_size = avifImageUsesU16(src.get()) ? 2 : 1;
    for (int plane = AVIF_CHAN_Y; plane <= AVIF_CHAN_A; ++plane) {
      uint8_t* row = avifImagePlane(src.get(), plane);
      ASSERT_NE(row, nullptr);
      const uint32_t row_bytes = avifImagePlaneRowBytes(src.get(), plane);
      for (uint32_t y = 0; y < 4; ++y, row += row_bytes) {
        for (uint32_t i = 0; i < 4 * pixel_size; ++i) {
          row[i] = static_cast<uint8_t>(plane * 64 + y * 16 + i);
        }
      }
    }
    // A view whose rows end at the last column of the source planes. Copying
    // it must only touch pointers within the source planes.
    ImagePtr view(avifImageCreateEmpty());
    ASSERT_NE(view, nullptr);
    const avifCropRect rect = {/*x=*/2, /*y=*/0, /*width=*/2, /*height=*/4};
    ASSERT_EQ(avifImageSetViewRect(view.get(), src.get(), &rect),
              AVIF_RESULT_OK);
    ImagePtr dst(avifImageCreateEmpty());
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(avifImageCopy(dst.get(), view.get(), AVIF_PLANES_ALL),
              AVIF_RESULT_OK);
    EXPECT_EQ(dst->width, 2u);
    EXPECT_EQ(dst->height, 4u);
    for (int plane = AVIF_CHAN_Y; plane <= AVIF_CHAN_A; ++plane) {
      const uint8_t* row = avifImagePlane(dst.get(), plane);
      ASSERT_NE(row, nullptr);
      const uint32_t row_bytes = avifImagePlaneRowBytes(dst.get(), plane);
      for (uint32_t y = 0; y < 4; ++y, row += row_bytes) {
        for (uint32_t i = 0; i < 2 * pixel_size; ++i) {
          EXPECT_EQ(row[i], static_cast<uint8_t>(plane * 64 + y * 16 +
                                                 2 * pixel_size + i));
        }
      }
    }
  }
}

TEST(ImageTest, CopyToItself) {
  ImagePtr image = testutil::CreateImage(/*width=*/4, /*height=*/4, /*depth=*/8,
                                         AVIF_PIXEL_FORMAT_YUV420,
                                         AVIF_PLANES_ALL, AVIF_RANGE_FULL);
  ASSERT_NE(image, nullptr);
  testutil::FillImageGradient(image.get(), /*offset=*/0);
  ImagePtr expected(avifImageCreateEmpty());
  ASSERT_NE(expected, nullptr);
  ASSERT_EQ(avifImageCopy(expected.get(), image.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);

  // An image cannot be copied to itself or be a view of itself.
  EXPECT_EQ(avifImageCopy(image.get(), image.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_INVALID_ARGUMENT);
  const avifCropRect rect = {/*x=*/0, /*y=*/0, /*width=*/2, /*height=*/2};
  EXPECT_EQ(avifImageSetViewRect(image.get(), image.get(), &rect),
            AVIF_RESULT_INVALID_ARGUMENT);
  // The image must be left untouched.
  EXPECT_TRUE(
      testutil::AreImagesEqual(*image, *expected, /*ignore_alpha=*/false));
}

TEST(ImageTest, IsOpaque) {
  for (uint32_t depth : {8, 10, 12, 16}) {
    SCOPED_TRACE(depth);
    // Odd dimensions to make sure the padding of the planes is not checked.
    ImagePtr image = testutil::CreateImage(
        /*width=*/3, /*height=*/3, depth, AVIF_PIXEL_FORMAT_YUV444,
        AVIF_PLANES_YUV, AVIF_RANGE_FULL);
    ASSERT_NE(image, nullptr);
    EXPECT_TRUE(avifImageIsOpaque(image.get()));

    ASSERT_EQ(avifImageAllocatePlanes(image.get(), AVIF_PLANES_A),
              AVIF_RESULT_OK);
    const uint32_t opaque_value = (1u << depth) - 1u;
    auto set_alpha = [&](uint32_t x, uint32_t y, uint32_t value) {
      uint8_t* row = image->alphaPlane + y * image->alphaRowBytes;
      if (avifImageUsesU16(image.get())) {
        reinterpret_cast<uint16_t*>(row)[x] = static_cast<uint16_t>(value);
      } else {
        row[x] = static_cast<uint8_t>(value);
      }
    };
    for (uint32_t y = 0; y < image->height; ++y) {
      for (uint32_t x = 0; x < image->width; ++x) {
        set_alpha(x, y, opaque_value);
      }
    }
    // An alpha plane with only opaque samples is opaque.
    EXPECT_TRUE(avifImageIsOpaque(image.get()));

    set_alpha(/*x=*/2, /*y=*/2, opaque_value - 1);
    EXPECT_FALSE(avifImageIsOpaque(image.get()));

    // A view excluding the non-opaque sample is opaque.
    ImagePtr view(avifImageCreateEmpty());
    ASSERT_NE(view, nullptr);
    const avifCropRect rect = {/*x=*/0, /*y=*/0, /*width=*/2, /*height=*/3};
    ASSERT_EQ(avifImageSetViewRect(view.get(), image.get(), &rect),
              AVIF_RESULT_OK);
    EXPECT_TRUE(avifImageIsOpaque(view.get()));
  }
}

TEST(ImageTest, SetViewRectEmptyRect) {
  ImagePtr src(avifImageCreate(2, 2, /*depth=*/8, AVIF_PIXEL_FORMAT_YUV444));
  ASSERT_NE(src, nullptr);
  ASSERT_EQ(avifImageAllocatePlanes(src.get(), AVIF_PLANES_ALL),
            AVIF_RESULT_OK);
  ImagePtr view(avifImageCreateEmpty());
  ASSERT_NE(view, nullptr);

  // Empty rects must be rejected. An empty rect on the bottom-right edge used
  // to produce plane pointers past the end of the source planes.
  const avifCropRect kEmptyRects[] = {
      {.x = 2, .y = 2, .width = 0, .height = 0},
      {.x = 0, .y = 0, .width = 0, .height = 0},
      {.x = 0, .y = 2, .width = 2, .height = 0},
      {.x = 2, .y = 0, .width = 0, .height = 2},
  };
  for (const avifCropRect& rect : kEmptyRects) {
    EXPECT_EQ(avifImageSetViewRect(view.get(), src.get(), &rect),
              AVIF_RESULT_INVALID_ARGUMENT)
        << rect.x << "," << rect.y << " " << rect.width << "x" << rect.height;
  }

  // The smallest valid rect on the bottom-right edge points to the last sample.
  const avifCropRect rect{.x = 1, .y = 1, .width = 1, .height = 1};
  ASSERT_EQ(avifImageSetViewRect(view.get(), src.get(), &rect), AVIF_RESULT_OK);
  EXPECT_EQ(view->width, 1u);
  EXPECT_EQ(view->height, 1u);
  for (int c = 0; c < 4; ++c) {
    EXPECT_EQ(avifImagePlane(view.get(), c),
              avifImagePlane(src.get(), c) +
                  avifImagePlaneRowBytes(src.get(), c) + 1);
  }
}

TEST(ImageTest, NullCases) {
  ImagePtr src(avifImageCreateEmpty());
  ImagePtr dst(avifImageCreateEmpty());
  // Both dst and src are nullptr.
  EXPECT_NE(avifImageCopy(nullptr, nullptr, AVIF_PLANES_ALL), AVIF_RESULT_OK);
  // src is nullptr.
  EXPECT_NE(avifImageCopy(dst.get(), nullptr, AVIF_PLANES_ALL), AVIF_RESULT_OK);
  // dst is nullptr.
  EXPECT_NE(avifImageCopy(nullptr, src.get(), AVIF_PLANES_ALL), AVIF_RESULT_OK);

  EXPECT_NE(avifImageAllocatePlanes(nullptr, AVIF_PLANES_ALL), AVIF_RESULT_OK);

  // This should not crash.
  avifImageFreePlanes(nullptr, AVIF_PLANES_ALL);

  // This should not crash.
  avifImageDestroy(nullptr);

  EXPECT_FALSE(avifImageUsesU16(nullptr));
  EXPECT_FALSE(avifImageIsOpaque(nullptr));
  EXPECT_EQ(avifImagePlane(nullptr, 0), nullptr);
  EXPECT_EQ(avifImagePlaneRowBytes(nullptr, 0), 0);
  EXPECT_EQ(avifImagePlaneWidth(nullptr, 0), 0);
  EXPECT_EQ(avifImagePlaneHeight(nullptr, 0), 0);

  // Various nullptr combinations for avifImageSetViewRect.
  EXPECT_NE(avifImageSetViewRect(nullptr, nullptr, nullptr), AVIF_RESULT_OK);

  EXPECT_NE(avifRWDataRealloc(nullptr, 10), AVIF_RESULT_OK);

  avifRWData rw_data;
  uint8_t raw_data[10] = {0};
  EXPECT_NE(avifRWDataSet(nullptr, nullptr, 10), AVIF_RESULT_OK);
  EXPECT_NE(avifRWDataSet(&rw_data, nullptr, 10), AVIF_RESULT_OK);
  EXPECT_NE(avifRWDataSet(nullptr, raw_data, 10), AVIF_RESULT_OK);
  EXPECT_NE(avifRWDataSet(nullptr, nullptr, 0), AVIF_RESULT_OK);

  // This should not crash.
  avifRWDataFree(nullptr);

  EXPECT_EQ(avifIOCreateMemoryReader(nullptr, 10), nullptr);
  EXPECT_EQ(avifIOCreateFileReader(nullptr), nullptr);

  // This should not crash.
  avifIODestroy(nullptr);
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
