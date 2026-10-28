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

//! Loop restoration stripe processing (spec 7.17, 7.17.1, 7.17.6). The
//! filters themselves live in `dsp::lr`.

use crate::decoder::av1::dsp::lr::LR_BORDER;
use crate::decoder::av1::headers::RestorationType;
use crate::decoder::av1::pixel::Pixel;

use super::buffers::PlaneWin;
use super::state::LrPlane;

/// Deblocked (pre-CDEF, upscaled) rows around one stripe boundary: rows
/// `b - 2, b - 1, b, b + 1` of every plane, where `b` is the first row of
/// stripe `k` in that plane.
pub struct LrBoundary<T> {
    pub k: usize,
    pub rows: [[Vec<T>; 4]; 3],
}

#[derive(Clone, Copy)]
pub struct StripeGeom {
    /// StripeStartY (plane rows, may be negative for the first stripe).
    pub start_y: isize,
    /// StripeEndY (inclusive).
    pub end_y: isize,
    pub plane_w: usize,
    pub plane_h: usize,
}

#[inline(always)]
fn copy_padded_row<T: Copy>(
    d: &mut [T],
    row: &[T],
    x0: usize,
    x1: usize,
    pw: usize,
    left: Option<&[T]>,
) {
    const B: usize = LR_BORDER;
    let (src_start, dst_start) = if let Some(l) = left {
        d[..B].copy_from_slice(l);
        (x0, B)
    } else if x0 >= B {
        (x0 - B, 0)
    } else {
        let pad_l = B - x0;
        d[..pad_l].fill(row[0]);
        (0, pad_l)
    };
    let end = (x1 + B).min(pw);
    let copy_len = end - src_start;
    d[dst_start..dst_start + copy_len].copy_from_slice(&row[src_start..end]);
    if x1 + B > pw {
        d[dst_start + copy_len..].fill(row[pw - 1]);
    }
}

/// Restores one stripe of one plane. Output rows
/// `max(0, start_y) ..= min(end_y, plane_h - 1)` are written to `out`
/// (row 0 = first output row).
#[allow(clippy::too_many_arguments)]
pub fn lr_stripe<T: Pixel>(
    lr: &LrPlane,
    g: &StripeGeom,
    cdef: &PlaneWin<T>,
    above: Option<&[Vec<T>; 4]>,
    below: Option<&[Vec<T>; 4]>,
    out: &mut [T],
    out_stride: usize,
    scratch: &mut Vec<T>,
    ss_y: usize,
    bd: u32,
    cols: (usize, usize),
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
) {
    let y_first = g.start_y.max(0) as usize;
    let y_last = (g.end_y as usize).min(g.plane_h - 1);
    let h = y_last + 1 - y_first;
    let pw = g.plane_w;
    let srows = h + 2 * LR_BORDER;
    let unit_size = lr.unit_size;
    let luma_off = 8 >> ss_y;
    let unit_row =
        (lr.unit_rows - 1).min(((g.start_y + luma_off as isize).max(0) as usize) / unit_size);
    for unit_col in 0..lr.unit_cols {
        let x0 = unit_col * unit_size;
        if x0 >= pw {
            break;
        }
        let x1 = if unit_col == lr.unit_cols - 1 {
            pw
        } else {
            ((unit_col + 1) * unit_size).min(pw)
        };
        let unit = &lr.units[unit_row * lr.unit_cols + unit_col];
        let w = x1 - x0;
        let filtered = x0 < cols.1
            && x1 > cols.0
            && matches!(unit.typ, RestorationType::Wiener | RestorationType::Sgrproj);
        if !filtered {
            // Unfiltered: the CDEF output.
            for i in 0..h {
                out[i * out_stride + x0..i * out_stride + x1]
                    .copy_from_slice(&cdef.row(y_first + i)[x0..x1]);
            }
            continue;
        }
        // Padded source of this unit only (columns x0 - LR_BORDER ..
        // x1 + LR_BORDER, replicating the first / last column of the
        // plane), small enough to stay in the L1 cache.
        let sstride = w + 2 * LR_BORDER;
        if scratch.len() < sstride * srows {
            scratch.resize(sstride * srows, T::default());
        }
        let src = &mut scratch[..sstride * srows];
        for (r, d) in src.chunks_exact_mut(sstride).enumerate() {
            let row = stripe_src_row(g, cdef, above, below, y_first, r);
            copy_padded_row(d, row, x0, x1, pw, None);
        }
        let dst = &mut out[x0..];
        match unit.typ {
            RestorationType::Wiener => {
                (dsp.wiener)(dst, out_stride, src, sstride, w, h, &unit.wiener, bd)
            }
            _ => (dsp.sgr)(
                dst,
                out_stride,
                src,
                sstride,
                w,
                h,
                unit.sgr_set as usize,
                unit.sgr_xqd,
                bd,
            ),
        }
    }
}

/// Source row of row `r` of the padded source of a stripe (stripe rows
/// plus `LR_BORDER` rows above and below, see [`lr_stripe`]).
#[inline]
fn stripe_src_row<'a, T: Pixel>(
    g: &StripeGeom,
    cdef: &'a PlaneWin<T>,
    above: Option<&'a [Vec<T>; 4]>,
    below: Option<&'a [Vec<T>; 4]>,
    y_first: usize,
    r: usize,
) -> &'a [T] {
    let mut y = y_first as isize + r as isize - LR_BORDER as isize;
    y = y.min(g.plane_h as isize - 1).max(0);
    if y < g.start_y {
        let y = y.max(g.start_y - 2);
        let idx = (y - (g.start_y - 2)) as usize; // 0 or 1
        match above {
            Some(a) => &a[idx][..],
            None => cdef.row((y as usize).max(cdef.y0)),
        }
    } else if y > g.end_y {
        let y = y.min(g.end_y + 2);
        let idx = (y - (g.end_y + 1)) as usize + 2; // 2 or 3
        match below {
            Some(b) => &b[idx][..],
            None => cdef.row(y as usize),
        }
    } else {
        cdef.row(y as usize)
    }
}

/// [`lr_stripe`] writing the restored samples back into `win` (rows
/// `max(0, start_y) ..= min(end_y, plane_h - 1)`), which holds the CDEF
/// output of the whole plane. Units are filtered left to right; the pre
/// restoration samples of the last `LR_BORDER` columns of a filtered
/// unit are kept in `left` for the next unit. Rows outside the stripe
/// come from `above` / `below` (or are clamped into the stripe at the
/// frame top and bottom), so they are never restored samples.
#[allow(clippy::too_many_arguments)]
pub fn lr_stripe_in_place<T: Pixel>(
    lr: &LrPlane,
    g: &StripeGeom,
    win: &mut PlaneWin<T>,
    above: Option<&[Vec<T>; 4]>,
    below: Option<&[Vec<T>; 4]>,
    scratch: &mut Vec<T>,
    left: &mut Vec<T>,
    ss_y: usize,
    bd: u32,
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
) {
    const B: usize = LR_BORDER;
    let y_first = g.start_y.max(0) as usize;
    let y_last = (g.end_y as usize).min(g.plane_h - 1);
    let h = y_last + 1 - y_first;
    let pw = g.plane_w;
    let srows = h + 2 * B;
    let unit_size = lr.unit_size;
    let luma_off = 8 >> ss_y;
    let unit_row =
        (lr.unit_rows - 1).min(((g.start_y + luma_off as isize).max(0) as usize) / unit_size);
    // x1 of the previous unit if it was restored (its last columns are in
    // `left`).
    let mut prev_x1 = usize::MAX;
    left.resize(srows * B, T::default());
    for unit_col in 0..lr.unit_cols {
        let x0 = unit_col * unit_size;
        if x0 >= pw {
            break;
        }
        let x1 = if unit_col == lr.unit_cols - 1 {
            pw
        } else {
            ((unit_col + 1) * unit_size).min(pw)
        };
        let unit = &lr.units[unit_row * lr.unit_cols + unit_col];
        if !matches!(unit.typ, RestorationType::Wiener | RestorationType::Sgrproj) {
            prev_x1 = usize::MAX;
            continue;
        }
        let w = x1 - x0;
        let sstride = w + 2 * B;
        if scratch.len() < sstride * srows {
            scratch.resize(sstride * srows, T::default());
        }
        let src = &mut scratch[..sstride * srows];
        let left_from_prev = prev_x1 == x0 && x0 >= B;
        for (r, d) in src.chunks_exact_mut(sstride).enumerate() {
            let row = stripe_src_row(g, win, above, below, y_first, r);
            let l = if left_from_prev { Some(&left[r * B..(r + 1) * B]) } else { None };
            copy_padded_row(d, row, x0, x1, pw, l);
        }
        // Pre restoration samples of columns x1 - B .. x1 for the next
        // unit, if the next unit is restored.
        let next_filtered = unit_col + 1 < lr.unit_cols
            && x1 < pw
            && matches!(
                lr.units[unit_row * lr.unit_cols + unit_col + 1].typ,
                RestorationType::Wiener | RestorationType::Sgrproj
            );
        if next_filtered {
            for (r, l) in left.chunks_exact_mut(B).enumerate() {
                l.copy_from_slice(&src[r * sstride + w..r * sstride + w + B]);
            }
            prev_x1 = x1;
        } else {
            prev_x1 = usize::MAX;
        }
        let stride = win.stride;
        let dst = win.slice_mut(x0, y_first);
        match unit.typ {
            RestorationType::Wiener => {
                (dsp.wiener)(dst, stride, src, sstride, w, h, &unit.wiener, bd)
            }
            _ => (dsp.sgr)(
                dst,
                stride,
                src,
                sstride,
                w,
                h,
                unit.sgr_set as usize,
                unit.sgr_xqd,
                bd,
            ),
        }
    }
}
