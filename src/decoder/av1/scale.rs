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

//! Streaming area-average (box) downscaler.
//!
//! Source rows are pushed top to bottom; each row is scaled horizontally
//! right away and accumulated into at most two pending output rows, so
//! the scaler only keeps O(output width) state. The result is identical
//! to scaling a whole image at once with [`scale_plane`].

/// Per output sample: index of the first source sample and the 16-bit
/// fixed point weights of the covered source samples (they sum to 1<<16).
#[derive(Clone, Debug)]
struct Taps {
    start: Vec<u32>,
    offsets: Vec<u32>,
    weights: Vec<u32>,
}

const WBITS: u32 = 16;

fn make_taps(src: usize, dst: usize) -> Taps {
    assert!(dst >= 1 && src >= dst);
    let mut start = Vec::with_capacity(dst);
    let mut offsets = Vec::with_capacity(dst + 1);
    let mut weights = Vec::new();
    let (s, d) = (src as u64, dst as u64);
    for x in 0..d {
        // Output sample x covers [x*s, (x+1)*s) in units of 1/d source.
        let lo = x * s;
        let hi = (x + 1) * s;
        let i0 = lo / d;
        let i1 = (hi - 1) / d;
        start.push(i0 as u32);
        offsets.push(weights.len() as u32);
        let mut sum = 0u64;
        for i in i0..=i1 {
            let a = lo.max(i * d);
            let b = hi.min((i + 1) * d);
            let w = if i == i1 { (1u64 << WBITS) - sum } else { ((b - a) << WBITS) / s };
            sum += w;
            weights.push(w as u32);
        }
    }
    offsets.push(weights.len() as u32);
    Taps {
        start,
        offsets,
        weights,
    }
}

/// Streaming scaler for one plane.
pub struct PlaneScaler {
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    h_taps: Taps,
    v_taps: Taps,
    /// Next source row expected.
    src_y: usize,
    /// Next output row to finish.
    dst_y: usize,
    acc: [Vec<u64>; 2],
    hrow: Vec<u32>,
}

impl PlaneScaler {
    pub fn new(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Self {
        Self {
            src_w,
            src_h,
            dst_w,
            dst_h,
            h_taps: make_taps(src_w, dst_w),
            v_taps: make_taps(src_h, dst_h),
            src_y: 0,
            dst_y: 0,
            acc: [vec![0; dst_w], vec![0; dst_w]],
            hrow: vec![0; dst_w],
        }
    }

    pub fn dst_size(&self) -> (usize, usize) {
        (self.dst_w, self.dst_h)
    }

    /// Pushes the next source row (`src_w` samples). Calls `emit(y, row)`
    /// for every output row that is complete.
    pub fn push_row<T: Copy + Into<u32>>(
        &mut self,
        row: &[T],
        mut emit: impl FnMut(usize, &[u32]),
    ) {
        debug_assert!(self.src_y < self.src_h);
        let ht = &self.h_taps;
        for x in 0..self.dst_w {
            let s = ht.start[x] as usize;
            let (o0, o1) = (ht.offsets[x] as usize, ht.offsets[x + 1] as usize);
            let mut v = 0u32;
            for (k, &w) in ht.weights[o0..o1].iter().enumerate() {
                v += w * row[s + k].into();
            }
            self.hrow[x] = v;
        }
        let y = self.src_y;
        // Output rows covering source row y: dst_y and possibly dst_y + 1.
        let mut out = [0usize; 2];
        let mut n = 0;
        for oy in self.dst_y..(self.dst_y + 2).min(self.dst_h) {
            let s = self.v_taps.start[oy] as usize;
            let cnt = (self.v_taps.offsets[oy + 1] - self.v_taps.offsets[oy]) as usize;
            if y >= s && y < s + cnt {
                let w = self.v_taps.weights[self.v_taps.offsets[oy] as usize + (y - s)] as u64;
                let acc = &mut self.acc[oy - self.dst_y];
                for (a, &h) in acc.iter_mut().zip(self.hrow.iter()) {
                    *a += w * h as u64;
                }
                out[n] = oy;
                n += 1;
            }
        }
        self.src_y += 1;
        // Finish output rows whose last source row is y.
        let mut tmp = vec![0u32; 0];
        for &oy in &out[..n] {
            let s = self.v_taps.start[oy] as usize;
            let cnt = (self.v_taps.offsets[oy + 1] - self.v_taps.offsets[oy]) as usize;
            if y + 1 == s + cnt {
                debug_assert_eq!(oy, self.dst_y);
                tmp.clear();
                tmp.extend(
                    self.acc[0]
                        .iter()
                        .map(|&a| ((a + (1u64 << (2 * WBITS - 1))) >> (2 * WBITS)) as u32),
                );
                emit(oy, &tmp);
                self.acc.swap(0, 1);
                self.acc[1].iter_mut().for_each(|a| *a = 0);
                self.dst_y += 1;
            }
        }
    }
}

/// Scales a whole plane (reference implementation for tests; same result
/// as streaming the rows through [`PlaneScaler`]).
pub fn scale_plane<T: Copy + Into<u32>>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    dw: usize,
    dh: usize,
) -> Vec<u32> {
    let mut s = PlaneScaler::new(w, h, dw, dh);
    let mut out = vec![0u32; dw * dh];
    for y in 0..h {
        s.push_row(&src[y * stride..y * stride + w], |oy, row| {
            out[oy * dw..(oy + 1) * dw].copy_from_slice(row)
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_average() {
        let src: Vec<u16> = (0..64).collect();
        let out = scale_plane(&src, 8, 8, 8, 8, 8);
        assert_eq!(out, (0..64).collect::<Vec<u32>>());
        let out = scale_plane(&src, 8, 8, 8, 4, 4);
        // 2x2 averages.
        for y in 0..4 {
            for x in 0..4 {
                let s = src[(2 * y) * 8 + 2 * x] as u32
                    + src[(2 * y) * 8 + 2 * x + 1] as u32
                    + src[(2 * y + 1) * 8 + 2 * x] as u32
                    + src[(2 * y + 1) * 8 + 2 * x + 1] as u32;
                assert_eq!(out[y * 4 + x], (s + 2) / 4);
            }
        }
        let flat = vec![200u8; 7 * 5];
        let out = scale_plane(&flat, 7, 7, 5, 3, 2);
        assert!(out.iter().all(|&v| v == 200));
    }
}
