// Copyright 2026 Google LLC
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

//! Horizontal super-resolution upscaling (spec 7.16), applied row by row.

use crate::decoder::av1::pixel::{clip_pixel, round2, Pixel};
use crate::decoder::av1::types::*;

const UPSCALE_FILTER: [[i32; 8]; 64] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];

/// Per plane upscaling parameters.
#[derive(Clone, Copy, Debug)]
pub struct Upscaler {
    pub upscaled_w: usize,
    step_x: i32,
    initial_subpel_x: i32,
    max_x: i32,
}

impl Upscaler {
    /// `frame_w` / `upscaled_w` are luma `FrameWidth` / `UpscaledWidth`.
    pub fn new(frame_w: usize, upscaled_w: usize, mi_cols: usize, sub_x: usize) -> Self {
        let down_w = ((frame_w + sub_x) >> sub_x) as i32;
        let up_w = ((upscaled_w + sub_x) >> sub_x) as i32;
        let step_x = ((down_w << SUPERRES_SCALE_BITS) + (up_w / 2)) / up_w;
        let err = up_w * step_x - (down_w << SUPERRES_SCALE_BITS);
        let mut initial = ((-((up_w - down_w) << (SUPERRES_SCALE_BITS - 1))) + up_w / 2) / up_w
            + (1 << (SUPERRES_EXTRA_BITS - 1))
            - err / 2;
        initial &= SUPERRES_SCALE_MASK as i32;
        let mi_w = (mi_cols >> sub_x) as i32;
        Self {
            upscaled_w: up_w as usize,
            step_x,
            initial_subpel_x: initial,
            max_x: mi_w * MI_SIZE as i32 - 1,
        }
    }

    /// Upscales one row.
    pub fn upscale_row<T: Pixel>(&self, dst: &mut [T], src: &[T], bd: u32) {
        for (x, d) in dst.iter_mut().take(self.upscaled_w).enumerate() {
            let src_x =
                -(1 << SUPERRES_SCALE_BITS) + self.initial_subpel_x + x as i32 * self.step_x;
            let src_x_px = src_x >> SUPERRES_SCALE_BITS;
            let subpel = ((src_x & SUPERRES_SCALE_MASK as i32) >> SUPERRES_EXTRA_BITS) as usize;
            let filter = &UPSCALE_FILTER[subpel];
            let mut sum = 0;
            for (k, &tap) in filter.iter().enumerate() {
                let sx = (src_x_px + k as i32 - 3).clamp(0, self.max_x) as usize;
                sum += src[sx].to_i32() * tap;
            }
            *d = clip_pixel(round2(sum, 7), bd);
        }
    }
}
