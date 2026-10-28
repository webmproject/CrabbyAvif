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

//! CDEF block iteration (spec 7.15, 7.15.1) for one post-filter step: a
//! range of mode info rows that starts and ends on 8x8 block rows (all of
//! a superblock row but its last 8 luma rows, plus the last 8 luma rows
//! of the row above, see `decode::Pipeline::post_row`). The direction
//! search and filter kernels live in `dsp::cdef`.
//!
//! The superblock row is processed in strips 64 luma samples wide (like
//! dav1d, which filters per 64x64 block): the padded pre-CDEF copy the
//! filter reads is built per strip, so it stays in the L1 cache instead
//! of being a copy of the whole superblock row. The pre-CDEF samples left
//! of a strip (already filtered in the frame) come from the previous
//! strip's copy. The copy holds pixels; samples outside the frame are not
//! written, the kernels derive their availability from the block
//! position. Every 64x64 unit of a strip (one `cdef_idx`) is then handed
//! to the `Dsp::cdef_unit` kernel, which runs the block loop.

use std::any::Any;

use crate::decoder::av1::dsp::cdef::{CdefPlane, CdefUnit, CDEF_BORDER, CDEF_SRC_SLACK};
use crate::decoder::av1::pixel::Pixel;

use super::buffers::{MiWin, PlaneWin};
use super::state::FrameCtx;

/// Padded pre-CDEF copy of one strip of one plane: plane columns
/// `x0 - CDEF_BORDER .. x1 + CDEF_BORDER` of rows
/// `y0 - CDEF_BORDER .. y1 + CDEF_BORDER` (only the samples inside the
/// frame are written).
pub struct CdefSrc {
    /// The copy: a `Vec<T>` of the decoder's pixel type `T`.
    buf: Box<dyn Any + Send + Sync>,
    pub stride: usize,
    /// Plane row stored at data row 0 (minus `CDEF_BORDER`).
    pub y0: usize,
    /// Plane column stored at data column 0 (minus `CDEF_BORDER`).
    pub x0: usize,
    /// Strip width `x1 - x0` of the previous strip built into `data`.
    prev_w: usize,
    /// Active plane row range `prev_ya0..prev_ya1` filtered by the previous strip.
    prev_ya0: usize,
    prev_ya1: usize,
}

impl CdefSrc {
    pub fn new() -> Self {
        Self {
            buf: Box::new(()),
            stride: 0,
            y0: 0,
            x0: 0,
            prev_w: 0,
            prev_ya0: 0,
            prev_ya1: 0,
        }
    }

    /// The copy as `Vec<T>` (emptied if it held another sample type).
    fn data_mut<T: Pixel>(&mut self) -> &mut Vec<T> {
        if !self.buf.is::<Vec<T>>() {
            self.buf = Box::new(Vec::<T>::new());
        }
        self.buf.downcast_mut::<Vec<T>>().unwrap()
    }

    fn data<T: Pixel>(&self) -> &[T] {
        self.buf.downcast_ref::<Vec<T>>().map_or(&[], |v| &v[..])
    }
}

impl Default for CdefSrc {
    fn default() -> Self {
        Self::new()
    }
}

/// Copies `N` samples.
#[inline(always)]
fn copy_n<T: Copy, const N: usize>(dst: &mut [T], src: &[T]) {
    let d: &mut [T; N] = (&mut dst[..N]).try_into().unwrap();
    let s: &[T; N] = src[..N].try_into().unwrap();
    *d = *s;
}

/// Fills one row of a strip copy (see [`build_strip`]): `line` is the
/// plane row, `drow` the copy row.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fill_strip_row<T: Pixel>(
    drow: &mut [T],
    line: &[T],
    x0: usize,
    x1: usize,
    w: usize,
    plane_w: usize,
    prev_w: usize,
    left_from_prev: bool,
) {
    const B: usize = CDEF_BORDER;
    if x0 > 0 {
        if left_from_prev {
            let l0 = drow[prev_w];
            let l1 = drow[prev_w + 1];
            drow[0] = l0;
            drow[1] = l1;
        } else {
            drow[0] = line[x0 - 2];
            drow[1] = line[x0 - 1];
        }
    }
    let r_end = (x1 + B).min(plane_w);
    let n = r_end - x0;
    if w == 64 && n == 64 + B {
        copy_n::<T, 66>(&mut drow[B..], &line[x0..]);
    } else if w == 32 && n == 32 + B {
        copy_n::<T, 34>(&mut drow[B..], &line[x0..]);
    } else {
        drow[B..B + n].copy_from_slice(&line[x0..r_end]);
    }
}

/// Fills `src` for active plane rows `ya0..ya1` (within superblock row
/// `y0..y1`) and columns `x0..x1` of `pl`.
/// `above` holds the two pre-CDEF rows above `y0` (if `y0 > 0`). Rows
/// `y1..y1 + 2` and the columns right of the strip are taken from `pl`
/// (not filtered yet) when inside the frame. The columns left of the
/// strip come from `pl` when not filtered by the previous strip, else
/// from the previous strip already in `src.data`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn build_strip<T: Pixel>(
    src: &mut CdefSrc,
    pl: &PlaneWin<T>,
    above: &[Vec<T>; 2],
    y0: usize,
    y1: usize,
    ya0: usize,
    ya1: usize,
    x0: usize,
    x1: usize,
    max_w: usize,
    plane_w: usize,
    plane_h: usize,
    has_left: bool,
) {
    const B: usize = CDEF_BORDER;
    let w = x1 - x0;
    let stride = max_w + 2 * B;
    let rows = y1 - y0 + 2 * B;
    let len = stride * rows + CDEF_SRC_SLACK;
    let prev_w = src.prev_w;
    let prev_ya0 = src.prev_ya0;
    let prev_ya1 = src.prev_ya1;
    let same_layout = src.stride == stride && src.data::<T>().len() == len;
    let can_left_from_prev = has_left && x0 >= B && prev_w >= B && same_layout;
    src.stride = stride;
    src.y0 = y0;
    src.x0 = x0;
    src.prev_w = w;
    src.prev_ya0 = ya0;
    src.prev_ya1 = ya1;
    let data = src.data_mut::<T>();
    // Every sample inside the frame is written below, so the buffer is
    // only resized.
    if data.len() != len {
        data.resize(len, T::default());
    }
    let r_start = ya0 - y0;
    let r_end = ya1 - y0 + 2 * B;
    for (r_off, drow) in data[r_start * stride..r_end * stride]
        .chunks_exact_mut(stride)
        .enumerate()
    {
        let r = r_start + r_off;
        let y = y0 as isize + r as isize - B as isize;
        if y < 0 || y as usize >= plane_h {
            continue;
        }
        let y = y as usize;
        let line: &[T] = if y < y0 { &above[y + 2 - y0][..] } else { pl.row(y) };
        let left_from_prev = can_left_from_prev && y >= prev_ya0 && y < prev_ya1;
        fill_strip_row(drow, line, x0, x1, w, plane_w, prev_w, left_from_prev);
    }
}

/// Strengths and filtered blocks of one 64x64 unit of a strip.
#[derive(Clone, Copy, Default)]
struct UnitInfo {
    /// First mode info row of the unit.
    mi_row: usize,
    noskip: [u8; 8],
    y_pri: i32,
    y_sec: i32,
    uv_pri: i32,
    uv_sec: i32,
}

impl UnitInfo {
    /// Luma is needed: filtered, or searched for the direction.
    fn need_y(&self) -> bool {
        self.y_pri != 0 || self.y_sec != 0 || self.uv_pri != 0
    }

    fn need_uv(&self) -> bool {
        self.uv_pri != 0 || self.uv_sec != 0
    }
}

/// Applies CDEF to mode-info rows `mi_row0..mi_row1` (both even). For
/// every plane `p`: `above[p]` are the two pre-CDEF rows above the range,
/// `rows[p]` its plane rows and `dims[p]` the plane size (width, height)
/// used for the CDEF borders.
#[allow(clippy::too_many_arguments)]
pub fn cdef_sbrow<T: Pixel>(
    f: &FrameCtx,
    mi: &MiWin,
    cdef_idx: impl Fn(usize, usize) -> i8,
    planes: &mut [PlaneWin<T>],
    srcs: &mut [CdefSrc],
    above: &[[Vec<T>; 2]],
    rows: &[(usize, usize)],
    dims: &[(usize, usize)],
    mi_row0: usize,
    mi_row1: usize,
    cols: (usize, usize),
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
) {
    let np = f.num_planes;
    let cd = &f.fh.cdef;
    let coeff_shift = f.bd - 8;
    let damping = cd.damping as i32 + coeff_shift as i32;
    let mi_row1 = mi_row1.min(f.mi_rows);
    let col_end = cols.1.min(f.mi_cols);
    let mut c0 = cols.0;
    let mut prev_filtered = [false; 3];
    while c0 < col_end {
        // Strips of 64 luma samples (16 mode info columns).
        let c1 = ((c0 | 15) + 1).min(col_end);
        // Units with filtered blocks, and the mode info row range needed
        // by luma and chroma. A step covers up to 3 (partial) 64x64 units
        // (128x128 superblocks: 2 + 16 + 14 mode info rows).
        let mut units = [UnitInfo::default(); 3];
        let mut n_units = 0;
        let mut rmin = [usize::MAX; 2];
        let mut rmax = [0usize; 2];
        let mut ur = mi_row0;
        while ur < mi_row1 {
            let ur_end = ((ur | 15) + 1).min(mi_row1);
            let idx = cdef_idx(ur, c0);
            if idx >= 0 {
                let idx = idx as usize;
                let mut u = UnitInfo {
                    mi_row: ur,
                    noskip: [0; 8],
                    y_pri: (cd.y_pri_strength[idx] as i32) << coeff_shift,
                    y_sec: (cd.y_sec_strength[idx] as i32) << coeff_shift,
                    uv_pri: if np > 1 {
                        (cd.uv_pri_strength[idx] as i32) << coeff_shift
                    } else {
                        0
                    },
                    uv_sec: if np > 1 {
                        (cd.uv_sec_strength[idx] as i32) << coeff_shift
                    } else {
                        0
                    },
                };
                let (need_y, need_uv) = (u.need_y(), u.need_uv());
                if need_y || need_uv {
                    let mut any = false;
                    let mut r = ur;
                    while r < ur_end {
                        let m0 = mi.row(r);
                        let m1 = mi.row(r + 1);
                        let mut mask = 0u8;
                        let mut bit = 1u8;
                        let mut c = c0;
                        while c < c1 {
                            if !(m0[c].skip && m1[c].skip && m0[c + 1].skip && m1[c + 1].skip) {
                                mask |= bit;
                            }
                            bit <<= 1;
                            c += 2;
                        }
                        if mask != 0 {
                            u.noskip[(r - ur) >> 1] = mask;
                            any = true;
                            for (k, need) in [need_y, need_uv].into_iter().enumerate() {
                                if need {
                                    rmin[k] = rmin[k].min(r);
                                    rmax[k] = r + 2;
                                }
                            }
                        }
                        r += 2;
                    }
                    if any {
                        units[n_units] = u;
                        n_units += 1;
                    }
                }
            }
            ur = ur_end;
        }
        if n_units == 0 {
            prev_filtered = [false; 3];
            c0 = c1;
            continue;
        }
        let mut built = [false; 3];
        for p in 0..np {
            let k = p.min(1);
            if rmin[k] >= rmax[k] {
                prev_filtered[p] = false;
                continue;
            }
            let (sx, sy) = f.sub(p);
            let (pw, ph) = dims[p];
            let (y0, y1) = rows[p];
            let ya0 = ((rmin[k] * 4) >> sy).max(y0);
            let ya1 = ((rmax[k] * 4) >> sy).min(y1);
            let x0 = (c0 * 4) >> sx;
            let x1 = ((c1 * 4) >> sx).min(pw);
            if x0 >= x1 || ya0 >= ya1 {
                prev_filtered[p] = false;
                continue;
            }
            build_strip(
                &mut srcs[p],
                &planes[p],
                &above[p],
                y0,
                y1,
                ya0,
                ya1,
                x0,
                x1,
                64 >> sx,
                pw,
                ph,
                prev_filtered[p],
            );
            prev_filtered[p] = true;
            built[p] = true;
        }
        for u in &units[..n_units] {
            let need = [u.need_y(), u.need_uv(), u.need_uv()];
            let mut views: [Option<CdefPlane<'_, T>>; 3] = [None, None, None];
            for (p, ((pl, src), view)) in planes
                .iter_mut()
                .zip(srcs.iter())
                .zip(views.iter_mut())
                .enumerate()
                .take(np)
            {
                if !(need[p] && built[p]) {
                    continue;
                }
                let (sx, sy) = f.sub(p);
                let (x0, y0) = ((c0 * 4) >> sx, (u.mi_row * 4) >> sy);
                *view = Some(CdefPlane {
                    dst_base: pl.idx(x0, y0),
                    dst_stride: pl.stride,
                    dst: pl.data_mut(),
                    src: src.data::<T>(),
                    src_stride: src.stride,
                    src_base: (y0 + CDEF_BORDER - src.y0) * src.stride
                        + (x0 + CDEF_BORDER - src.x0),
                    x0,
                    y0,
                    plane_w: dims[p].0,
                    plane_h: dims[p].1,
                });
            }
            let mut unit = CdefUnit {
                planes: views,
                noskip: u.noskip,
                y_pri: u.y_pri,
                y_sec: u.y_sec,
                uv_pri: u.uv_pri,
                uv_sec: u.uv_sec,
                damping,
                ss_x: f.ss_x,
                ss_y: f.ss_y,
                bd: f.bd,
            };
            (dsp.cdef_unit)(&mut unit);
        }
        c0 = c1;
    }
}
