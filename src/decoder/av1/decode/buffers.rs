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
/// The samples are a plain `Vec` (`data()[0]` is sample (x0, y0)); its
/// start has no alignment guarantee beyond that of `T`.
#[derive(Clone, Default)]
pub struct PlaneWin<T> {
    buf: Vec<T>,
    pub stride: usize,
    pub x0: usize,
    pub y0: usize,
    pub rows: usize,
}

impl<T: Pixel> PlaneWin<T> {
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn new(stride: usize, rows: usize) -> Self {
        Self::new_tile(stride, rows, 0, 0)
    }

    pub fn from_vec(mut buf: Vec<T>, stride: usize, rows: usize) -> Self {
        Self::fit(&mut buf, stride * rows);
        Self {
            buf,
            stride,
            x0: 0,
            y0: 0,
            rows,
        }
    }

    /// Resizes `buf` to `n` samples. A buffer that has to grow is
    /// replaced by a zeroed allocation (`calloc`: fresh pages are zero
    /// already) instead of being filled, which for frame sized planes
    /// would write (and fault in) every page once more before decoding.
    fn fit(buf: &mut Vec<T>, n: usize) {
        if buf.capacity() < n {
            *buf = vec![T::default(); n];
        } else {
            buf.resize(n, T::default());
        }
    }

    pub fn new_tile(stride: usize, rows: usize, x0: usize, y0: usize) -> Self {
        let buf = vec![T::default(); stride * rows];
        Self {
            buf,
            stride,
            x0,
            y0,
            rows,
        }
    }

    pub fn reset(&mut self, stride: usize, rows: usize, x0: usize, y0: usize) {
        Self::fit(&mut self.buf, stride * rows);
        self.stride = stride;
        self.x0 = x0;
        self.y0 = y0;
        self.rows = rows;
    }

    /// The samples: `data()[0]` is sample (x0, y0), rows `stride` apart.
    #[inline(always)]
    pub fn data(&self) -> &[T] {
        &self.buf
    }

    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.buf
    }

    /// The allocation and the offset of sample (x0, y0) in it (for
    /// handing the plane over without a copy). The offset is always 0
    /// (kept so that `sink::Plane::offset` stays meaningful).
    pub fn into_raw(self) -> (Vec<T>, usize) {
        (self.buf, 0)
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
        debug_assert!(self.stride <= 0x3_ffff);
        // Each term is masked to 18 bits: AV1 planes are at most 65536
        // samples wide / high, so window rows, columns and strides (plus
        // padding) always fit, and the masks are no-ops for valid
        // coordinates. They let the compiler prove that the index
        // arithmetic cannot overflow, which keeps it simple in the
        // inlined hot loops. Out-of-window coordinates are caught by the
        // debug assertions above (in release builds they alias to another
        // in-bounds index or fail the slice bounds check).
        ((y - self.y0) & 0x3_ffff) * (self.stride & 0x3_ffff) + ((x - self.x0) & 0x3_ffff)
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.buf[self.idx(x, y)]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        let i = self.idx(x, y);
        self.buf[i] = v;
    }

    /// Slice starting at sample (x, y) (rows continue with `stride`).
    #[inline(always)]
    pub fn slice_mut(&mut self, x: usize, y: usize) -> &mut [T] {
        let i = self.idx(x, y);
        &mut self.buf[i..]
    }

    #[inline(always)]
    pub fn slice(&self, x: usize, y: usize) -> &[T] {
        let i = self.idx(x, y);
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
        let i = (y - self.y0) * self.stride;
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
        let i = (y - self.y0) * self.stride;
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
            let s = self.stride;
            self.buf.copy_within(shift * s..(shift + keep) * s, 0);
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
#[derive(Clone, Default)]
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
        let mut w = Self::default();
        w.reset(
            stride, rows, col0, y0, ss_x, ss_y, deblock_y, deblock_uv, delta_lf, intrabc,
        );
        w
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reset(
        &mut self,
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
    ) {
        debug_assert!(rows.is_power_of_two());
        let n = stride * rows;
        let uv_stride = stride >> ss_x;
        let uv_rows = rows >> ss_y;
        let uv_n = uv_stride * uv_rows;
        if intrabc {
            let pl = self.data.len().min(n);
            self.data[..pl].fill(MiInfo::default());
        }
        self.data.resize(n, MiInfo::default());
        if deblock_y {
            if self.lf_tx[0].capacity() < n {
                self.lf_tx[0] = vec![0; n];
            } else {
                self.lf_tx[0].resize(n, 0);
            }
        } else {
            self.lf_tx[0].clear();
        }
        if deblock_uv {
            if self.lf_tx[1].capacity() < uv_n {
                self.lf_tx[1] = vec![0; uv_n];
            } else {
                self.lf_tx[1].resize(uv_n, 0);
            }
        } else {
            self.lf_tx[1].clear();
        }
        self.lf_stride = [stride, uv_stride];
        self.lf_mask = [rows - 1, uv_rows - 1];
        if delta_lf {
            self.delta_lf.resize(n, [0; 4]);
        } else {
            self.delta_lf.clear();
        }
        if intrabc {
            let pl = self.mv.len().min(n);
            self.mv[..pl].fill([0; 2]);
            self.mv.resize(n, [0; 2]);
        } else {
            self.mv.clear();
        }
        self.stride = stride;
        self.col0 = col0;
        self.y0 = y0;
        self.rows = rows;
        self.row_mask = rows - 1;
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
