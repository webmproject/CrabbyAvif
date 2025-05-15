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
#include "testutil.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <vector>

#include "avif/avif.h"
#include "avif/libavif_compat.h"
#include "gtest/gtest.h"

using namespace crabbyavif;

namespace testutil {

std::vector<uint8_t> read_file(const char* file_name) {
  std::ifstream file(file_name, std::ios::binary);
  EXPECT_TRUE(file.is_open());
  // Get file size.
  file.seekg(0, std::ios::end);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  file.read(reinterpret_cast<char*>(data.data()), size);
  file.close();
  return data;
}

avif::ImagePtr CreateImage(int width, int height, int depth,
                           avifPixelFormat yuv_format, avifPlanesFlags planes,
                           avifRange yuv_range) {
  avif::ImagePtr image(avifImageCreate(width, height, depth, yuv_format));
  if (!image) {
    return nullptr;
  }
  image->yuvRange = yuv_range;
  if (avifImageAllocatePlanes(image.get(), planes) != AVIF_RESULT_OK) {
    return nullptr;
  }
  return image;
}

void FillImageGradient(avifImage* image, int offset) {
  for (avifChannelIndex c :
       {AVIF_CHAN_Y, AVIF_CHAN_U, AVIF_CHAN_V, AVIF_CHAN_A}) {
    const uint32_t limitedRangeMin =
        c == AVIF_CHAN_Y ? 16 << (image->depth - 8) : 0;
    const uint32_t limitedRangeMax = (c == AVIF_CHAN_Y ? 219 : 224)
                                     << (image->depth - 8);

    const uint32_t plane_width = avifImagePlaneWidth(image, c);
    // 0 for A if no alpha and 0 for UV if 4:0:0.
    const uint32_t plane_height = avifImagePlaneHeight(image, c);
    uint8_t* row = avifImagePlane(image, c);
    const uint32_t row_bytes = avifImagePlaneRowBytes(image, c);
    const uint32_t max_xy_sum = plane_width + plane_height - 2;
    for (uint32_t y = 0; y < plane_height; ++y) {
      for (uint32_t x = 0; x < plane_width; ++x) {
        uint32_t value = (x + y + offset) % (max_xy_sum + 1);
        if (image->yuvRange == AVIF_RANGE_FULL || c == AVIF_CHAN_A) {
          value =
              value * ((1u << image->depth) - 1u) / std::max(1u, max_xy_sum);
        } else {
          value = limitedRangeMin + value *
                                        (limitedRangeMax - limitedRangeMin) /
                                        std::max(1u, max_xy_sum);
        }
        if (avifImageUsesU16(image)) {
          reinterpret_cast<uint16_t*>(row)[x] = static_cast<uint16_t>(value);
        } else {
          row[x] = static_cast<uint8_t>(value);
        }
      }
      row += row_bytes;
    }
  }
}

bool AreByteSequencesEqual(const uint8_t* data1, size_t data1_length,
                           const uint8_t* data2, size_t data2_length) {
  if (data1_length != data2_length) return false;
  return data1_length == 0 || std::equal(data1, data1 + data1_length, data2);
}

}  // namespace testutil
