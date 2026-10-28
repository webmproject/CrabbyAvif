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

//! Pixel and mode-info storage for the superblock row pipeline.
//!
//! Both pixel planes and the 4x4 mode info grid only hold a sliding window
//! of rows (a few superblock rows), never the whole frame (except for
//! frames using intra block copy, where the window covers the frame).

use crate::decoder::av1::pixel::Pixel;

/// A window of rows of one plane. Row `y` (absolute, in plane samples) is
/// stored at data row `y - y0`.
///
/// The sample storage starts at a 64-byte aligned address (`data()` is
/// an offset view into a slightly larger allocation), so with strides
/// that are multiples of 64 bytes every row starts on a cache line and
/// full-width vector stores of 32 / 64 sample rows do not split lines.
#[derive(Clone)]
pub struct PlaneWin<T> {
    buf: Vec<T>,
    /// Elements of `buf` before the first (64-byte aligned) sample.
    off: usize,
    pub stride: usize,
    pub x0: usize,
    pub y0: usize,
    pub rows: usize,
}

/// Allocates `n` default samples starting at a 64-byte aligned address:
/// returns the allocation and the offset (in elements) of that start.
fn alloc_aligned<T: Pixel>(n: usize) -> (Vec<T>, usize) {
    let sz = std::mem::size_of::<T>();
    let buf = vec![T::default(); n + 64 / sz];
    let addr = buf.as_ptr() as usize;
    let off = ((64 - addr % 64) % 64) / sz;
    (buf, off)
}

impl<T: Pixel> PlaneWin<T> {
    pub fn new(stride: usize, rows: usize) -> Self {
        Self::new_tile(stride, rows, 0, 0)
    }

    pub fn new_tile(stride: usize, rows: usize, x0: usize, y0: usize) -> Self {
        let (buf, off) = alloc_aligned(stride * rows);
        Self {
            buf,
            off,
            stride,
            x0,
            y0,
            rows,
        }
    }

    /// The samples: `data()[0]` is sample (x0, y0), rows `stride` apart.
    #[inline(always)]
    pub fn data(&self) -> &[T] {
        &self.buf[self.off..]
    }

    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.buf[self.off..]
    }

    /// The allocation and the offset of sample (x0, y0) in it (for
    /// handing the plane over without a copy).
    pub fn into_raw(self) -> (Vec<T>, usize) {
        (self.buf, self.off)
    }

    #[inline(always)]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        debug_assert!(
            y >= self.y0 && y < self.y0 + self.rows,
            "row {y} not in window {}+{}",
            self.y0,
            self.rows
        );
        debug_assert!(
            x >= self.x0 && x < self.x0 + self.stride,
            "col {x} not in window {}+{}",
            self.x0,
            self.stride
        );
        (y - self.y0) * self.stride + (x - self.x0)
    }

    // The accessors below index `buf` with `off` folded into the index
    // (one add) instead of slicing `data()` first (an extra bounds check).

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.buf[self.off + self.idx(x, y)]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        let i = self.off + self.idx(x, y);
        self.buf[i] = v;
    }

    /// Slice starting at sample (x, y) (rows continue with `stride`).
    #[inline(always)]
    pub fn slice_mut(&mut self, x: usize, y: usize) -> &mut [T] {
        let i = self.off + self.idx(x, y);
        &mut self.buf[i..]
    }

    #[inline(always)]
    pub fn slice(&self, x: usize, y: usize) -> &[T] {
        let i = self.off + self.idx(x, y);
        &self.buf[i..]
    }

    /// Row `y` as a slice of `stride` samples.
    #[inline(always)]
    pub fn row(&self, y: usize) -> &[T] {
        debug_assert!(
            y >= self.y0 && y < self.y0 + self.rows,
            "row {y} not in window {}+{}",
            self.y0,
            self.rows
        );
        let i = self.off + (y - self.y0) * self.stride;
        &self.buf[i..i + self.stride]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        debug_assert!(
            y >= self.y0 && y < self.y0 + self.rows,
            "row {y} not in window {}+{}",
            self.y0,
            self.rows
        );
        let i = self.off + (y - self.y0) * self.stride;
        &mut self.buf[i..i + self.stride]
    }

    #[inline(always)]
    pub fn contains(&self, y: usize) -> bool {
        y >= self.y0 && y < self.y0 + self.rows
    }

    /// Makes the window cover rows `new_y0..need_end` (`new_y0` >= the
    /// current y0), keeping the content of rows `new_y0..keep_end`. The
    /// window only moves when `need_end` is not inside it yet, so a
    /// window with spare rows is moved (copied) less often.
    pub fn slide_to(&mut self, new_y0: usize, need_end: usize, keep_end: usize) {
        if new_y0 <= self.y0 || need_end <= self.y0 + self.rows {
            return;
        }
        debug_assert!(need_end - new_y0 <= self.rows);
        let shift = new_y0 - self.y0;
        let keep = keep_end
            .saturating_sub(new_y0)
            .min(self.rows.saturating_sub(shift));
        if keep > 0 {
            let (s, o) = (self.stride, self.off);
            self.buf
                .copy_within(o + shift * s..o + (shift + keep) * s, o);
        }
        self.y0 = new_y0;
    }
}

/// Mode info stored for every 4x4 luma block (spec per-position arrays
/// such as MiSizes, YModes, Skips, ...).
#[derive(Clone, Copy, Default, Debug)]
#[repr(C, align(8))]
pub struct MiInfo {
    pub bsize: u8,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub skip: bool,
    pub is_inter: bool,
    pub segment_id: u8,
    /// `InterTxSizes`.
    pub inter_tx_size: u8,
    /// `PaletteSizes[0]` in bits 0..4, `PaletteSizes[1]` in bits 4..8.
    pub palette_sizes: u8,
}

impl MiInfo {
    #[inline(always)]
    pub fn palette_size(&self, plane: usize) -> usize {
        ((self.palette_sizes >> (plane * 4)) & 0xf) as usize
    }
}

/// Window of mode info rows (in 4x4 units). `rows` is a power of two
/// (`2 * sb4`), so rows wrap in place without copying on `slide_to`.
#[derive(Clone)]
pub struct MiWin {
    pub data: Vec<MiInfo>,
    /// `LoopfilterTxSizes` for luma (`0`) and chroma (`1`, subsampled
    /// by `(ss_x, ss_y)`), empty when deblocking is off.
    pub lf_tx: [Vec<u8>; 2],
    pub lf_stride: [usize; 2],
    pub lf_mask: [usize; 2],
    /// `DeltaLFs` per luma 4x4 block (empty unless `delta_lf_present`).
    pub delta_lf: Vec<[i8; 4]>,
    /// `Mvs[..][0]` per luma 4x4 block (empty unless `allow_intrabc`).
    pub mv: Vec<[i16; 2]>,
    pub stride: usize,
    pub col0: usize,
    pub y0: usize,
    pub rows: usize,
    pub row_mask: usize,
}

impl MiWin {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stride: usize,
        rows: usize,
        ss_x: usize,
        ss_y: usize,
        deblock_y: bool,
        deblock_uv: bool,
        delta_lf: bool,
        intrabc: bool,
    ) -> Self {
        Self::new_tile(
            stride, rows, 0, 0, ss_x, ss_y, deblock_y, deblock_uv, delta_lf, intrabc,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_tile(
        stride: usize,
        rows: usize,
        col0: usize,
        y0: usize,
        ss_x: usize,
        ss_y: usize,
        deblock_y: bool,
        deblock_uv: bool,
        delta_lf: bool,
        intrabc: bool,
    ) -> Self {
        debug_assert!(rows.is_power_of_two());
        let n = stride * rows;
        let uv_stride = stride >> ss_x;
        let uv_rows = rows >> ss_y;
        Self {
            data: vec![MiInfo::default(); n],
            lf_tx: [
                if deblock_y { vec![0; n] } else { Vec::new() },
                if deblock_uv { vec![0; uv_stride * uv_rows] } else { Vec::new() },
            ],
            lf_stride: [stride, uv_stride],
            lf_mask: [rows - 1, uv_rows - 1],
            delta_lf: if delta_lf { vec![[0; 4]; n] } else { Vec::new() },
            mv: if intrabc { vec![[0; 2]; n] } else { Vec::new() },
            stride,
            col0,
            y0,
            rows,
            row_mask: rows - 1,
        }
    }

    #[inline(always)]
    pub fn idx(&self, row: usize, col: usize) -> usize {
        debug_assert!(
            row >= self.y0 && row < self.y0 + self.rows,
            "mi row {row} not in {}+{}",
            self.y0,
            self.rows
        );
        debug_assert!(
            col >= self.col0 && col < self.col0 + self.stride,
            "mi col {col} not in {}+{}",
            self.col0,
            self.stride
        );
        (row & self.row_mask) * self.stride + (col - self.col0)
    }

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> &MiInfo {
        &self.data[self.idx(row, col)]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut MiInfo {
        let i = self.idx(row, col);
        &mut self.data[i]
    }

    #[inline(always)]
    pub fn mv_at(&self, row: usize, col: usize) -> [i16; 2] {
        self.mv[self.idx(row, col)]
    }

    #[inline(always)]
    pub fn delta_lf_at(&self, row: usize, col: usize) -> [i8; 4] {
        self.delta_lf[self.idx(row, col)]
    }

    pub fn contains(&self, row: usize) -> bool {
        row >= self.y0 && row < self.y0 + self.rows
    }

    /// Mode info row `row` (`stride` entries).
    #[inline(always)]
    pub fn row(&self, row: usize) -> &[MiInfo] {
        debug_assert!(
            row >= self.y0 && row < self.y0 + self.rows,
            "mi row {row} not in {}+{}",
            self.y0,
            self.rows
        );
        let i = (row & self.row_mask) * self.stride;
        &self.data[i..i + self.stride]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, row: usize) -> &mut [MiInfo] {
        debug_assert!(
            row >= self.y0 && row < self.y0 + self.rows,
            "mi row {row} not in {}+{}",
            self.y0,
            self.rows
        );
        let i = (row & self.row_mask) * self.stride;
        &mut self.data[i..i + self.stride]
    }

    #[inline(always)]
    pub fn lf_tx_row(&self, pt: usize, r4: usize) -> &[u8] {
        let s = self.lf_stride[pt];
        let i = (r4 & self.lf_mask[pt]) * s;
        &self.lf_tx[pt][i..i + s]
    }

    #[inline(always)]
    pub fn lf_tx_row_mut(&mut self, pt: usize, r4: usize) -> &mut [u8] {
        let s = self.lf_stride[pt];
        let i = (r4 & self.lf_mask[pt]) * s;
        &mut self.lf_tx[pt][i..i + s]
    }

    pub fn slide_to(&mut self, new_y0: usize) {
        if new_y0 > self.y0 {
            self.y0 = new_y0;
        }
    }
}
