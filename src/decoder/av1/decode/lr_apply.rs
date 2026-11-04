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

use crate::decoder::av1::dsp::lr::{LR_BORDER, LR_MAX_H, LR_MAX_W, LR_SRC_SLACK};
use crate::decoder::av1::dsp::simd::lr::sgr_filter_units;
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
fn copy_padded_row<T: Copy>(d: &mut [T], row: &[T], x0: usize, x1: usize, pw: usize) {
    const B: usize = LR_BORDER;
    let (src_start, dst_start) = if x0 >= B {
        (x0 - B, 0)
    } else {
        let pad_l = B - x0;
        let v = row[0];
        for x in &mut d[..pad_l] {
            *x = v;
        }
        (0, pad_l)
    };
    let end = (x1 + B).min(pw);
    let copy_len = end - src_start;
    d[dst_start..dst_start + copy_len].copy_from_slice(&row[src_start..end]);
    if x1 + B > pw {
        let v = row[pw - 1];
        for x in &mut d[dst_start + copy_len..] {
            *x = v;
        }
    }
}

#[inline(always)]
fn flush_unit_rows<T: Copy, const W: usize>(mut dst: &mut [T], stride: usize, src: &[T], h: usize) {
    let src = &src[..h * W];
    for i in 0..h - 1 {
        dst[..W].copy_from_slice(&src[i * W..(i + 1) * W]);
        dst = &mut dst[stride..];
    }
    dst[..W].copy_from_slice(&src[(h - 1) * W..h * W]);
}

#[inline(always)]
fn flush_prev<T: Pixel>(
    win: &mut PlaneWin<T>,
    prev_buf: &[T],
    y_first: usize,
    h: usize,
    px0: usize,
    pw_u: usize,
) {
    let stride = win.stride;
    let dst_base = (y_first - win.y0) * stride + px0;
    let dst_data = &mut win.data_mut()[dst_base..dst_base + (h - 1) * stride + pw_u];
    match pw_u {
        32 => flush_unit_rows::<T, 32>(dst_data, stride, prev_buf, h),
        64 => flush_unit_rows::<T, 64>(dst_data, stride, prev_buf, h),
        128 => flush_unit_rows::<T, 128>(dst_data, stride, prev_buf, h),
        256 => flush_unit_rows::<T, 256>(dst_data, stride, prev_buf, h),
        _ => {
            let src_data = &prev_buf[..h * pw_u];
            for i in 0..h {
                dst_data[i * stride..i * stride + pw_u]
                    .copy_from_slice(&src_data[i * pw_u..(i + 1) * pw_u]);
            }
        }
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
    let cdef_stride = cdef.stride;
    let cdef_base = (y_first - cdef.y0) * cdef_stride;
    let cdef_data = &cdef.data()[cdef_base..];
    let unit_x1 = |c: usize| if c == lr.unit_cols - 1 { pw } else { ((c + 1) * unit_size).min(pw) };
    let mut unit_col = 0;
    while unit_col < lr.unit_cols {
        let x0 = unit_col * unit_size;
        if x0 >= pw {
            break;
        }
        let x1 = unit_x1(unit_col);
        let unit = &lr.units[unit_row * lr.unit_cols + unit_col];
        let w = x1 - x0;
        let filtered = x0 < cols.1
            && x1 > cols.0
            && matches!(unit.typ, RestorationType::Wiener | RestorationType::Sgrproj);
        unit_col += 1;
        if !filtered {
            // Unfiltered: coalesce any run of adjacent unfiltered units into a
            // single row-by-row (or whole-stripe) copy of the CDEF output.
            let mut span_x1 = x1;
            while unit_col < lr.unit_cols {
                let nx0 = unit_col * unit_size;
                if nx0 >= pw {
                    break;
                }
                let nx1 = unit_x1(unit_col);
                let nu = &lr.units[unit_row * lr.unit_cols + unit_col];
                let nfiltered = nx0 < cols.1
                    && nx1 > cols.0
                    && matches!(nu.typ, RestorationType::Wiener | RestorationType::Sgrproj);
                if nfiltered {
                    break;
                }
                span_x1 = nx1;
                unit_col += 1;
            }
            if x0 == 0 && span_x1 == pw && out_stride == pw && cdef_stride == pw {
                out[..h * pw].copy_from_slice(&cdef_data[..h * pw]);
            } else {
                for i in 0..h {
                    out[i * out_stride + x0..i * out_stride + span_x1].copy_from_slice(
                        &cdef_data[i * cdef_stride + x0..i * cdef_stride + span_x1],
                    );
                }
            }
            continue;
        }
        // Padded source rows of this unit (columns x0 - LR_BORDER ..
        // x1 + LR_BORDER). Away from the plane's left and right edges they
        // are the source rows themselves (which extend past the unit, see
        // LR_SRC_SLACK); at the edges the first / last column is
        // replicated into a copy of the unit, small enough to stay in the
        // L1 cache.
        let mut rows: [&[T]; LR_MAX_H + 2 * LR_BORDER] = [&[]; LR_MAX_H + 2 * LR_BORDER];
        if x0 >= LR_BORDER && x1 + LR_BORDER <= pw {
            let off = x0 - LR_BORDER;
            for (r, d) in rows[..LR_BORDER].iter_mut().enumerate() {
                *d = &stripe_src_row(g, cdef, above, below, y_first, r)[off..];
            }
            let mut win_rows = &cdef_data[off..];
            if let Some((last, head)) = rows[LR_BORDER..LR_BORDER + h].split_last_mut() {
                for d in head {
                    *d = win_rows;
                    win_rows = &win_rows[cdef_stride..];
                }
                *last = win_rows;
            }
            for (r, d) in rows[..srows].iter_mut().enumerate().skip(LR_BORDER + h) {
                *d = &stripe_src_row(g, cdef, above, below, y_first, r)[off..];
            }
            // A run of interior self guided units with the same parameter set
            // is filtered in one call (box sums and `A` / `B` are shared).
            if matches!(unit.typ, RestorationType::Sgrproj)
                && w % 16 == 0
                && unit_col < lr.unit_cols
            {
                let can_merge = |uc: usize, cur_w: usize| -> Option<(usize, [i32; 2])> {
                    if cur_w % 16 != 0 {
                        return None;
                    }
                    let cx0 = uc * unit_size;
                    if cx0 >= pw {
                        return None;
                    }
                    let cx1 = unit_x1(uc);
                    let u = &lr.units[unit_row * lr.unit_cols + uc];
                    if matches!(u.typ, RestorationType::Sgrproj)
                        && u.sgr_set == unit.sgr_set
                        && cx0 < cols.1
                        && cx1 > cols.0
                        && cx1 + LR_BORDER <= pw
                        && cx1 - x0 <= LR_MAX_W
                    {
                        Some((cx1, u.sgr_xqd))
                    } else {
                        None
                    }
                };
                if let Some((mut gx1, second_xqd)) = can_merge(unit_col, w) {
                    let mut units = [(0usize, [0i32; 2]); LR_MAX_W / 16];
                    units[0] = (w, unit.sgr_xqd);
                    units[1] = (gx1 - x1, second_xqd);
                    let mut n = 2;
                    unit_col += 1;
                    while unit_col < lr.unit_cols && n < units.len() {
                        let Some((cx1, xqd)) = can_merge(unit_col, units[n - 1].0) else {
                            break;
                        };
                        units[n] = (cx1 - gx1, xqd);
                        n += 1;
                        gx1 = cx1;
                        unit_col += 1;
                    }
                    sgr_filter_units(
                        &mut out[x0..],
                        out_stride,
                        &rows[..srows],
                        gx1 - x0,
                        h,
                        unit.sgr_set as usize,
                        &units[..n],
                        bd,
                    );
                    continue;
                }
            }
        } else {
            let sstride = w + 2 * LR_BORDER;
            let len = sstride * srows + LR_SRC_SLACK;
            if scratch.len() < len {
                scratch.resize(len, T::default());
            }
            for r in 0..srows {
                let row = stripe_src_row(g, cdef, above, below, y_first, r);
                copy_padded_row(
                    &mut scratch[r * sstride..(r + 1) * sstride],
                    row,
                    x0,
                    x1,
                    pw,
                );
            }
            for (r, d) in rows[..srows].iter_mut().enumerate() {
                *d = &scratch[r * sstride..(r + 1) * sstride + LR_SRC_SLACK];
            }
        }
        let src = &rows[..srows];
        let dst = &mut out[x0..];
        match unit.typ {
            RestorationType::Wiener => (dsp.wiener)(dst, out_stride, src, w, h, &unit.wiener, bd),
            _ => (dsp.sgr)(
                dst,
                out_stride,
                src,
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
/// output of the whole plane. Units are filtered left to right; since a
/// unit reads the un-restored last `LR_BORDER` columns of the unit to its
/// left, the restored output of a filtered unit is kept in `scratch` and
/// written to `win` only after the next unit has read its source rows.
/// Rows outside the stripe come from `above` / `below` (or are clamped
/// into the stripe at the frame top and bottom), so they are never
/// restored samples. `_left` is unused.
#[allow(clippy::too_many_arguments)]
pub fn lr_stripe_in_place<T: Pixel>(
    lr: &LrPlane,
    g: &StripeGeom,
    win: &mut PlaneWin<T>,
    above: Option<&[Vec<T>; 4]>,
    below: Option<&[Vec<T>; 4]>,
    scratch: &mut Vec<T>,
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

    // The last unit of a row extends to the plane's right edge, so it can
    // be up to about 1.5 times `unit_size` wide (see
    // `count_units_in_frame`); it always takes the edge path below.
    let last_w = pw.saturating_sub(lr.unit_cols.saturating_sub(1) * unit_size);
    let max_sstride = unit_size.max(last_w) + 2 * B;
    let edge_len = max_sstride * srows + LR_SRC_SLACK;
    // Only units followed by another unit are delayed (never the last one),
    // so these are at most `unit_size` wide.
    let out_len = unit_size * h;
    let total_len = edge_len + 2 * out_len;
    if scratch.len() < total_len {
        scratch.resize(total_len, T::default());
    }
    let (edge_scratch, out_both) = scratch[..total_len].split_at_mut(edge_len);
    let (mut out_prev, mut out_cur) = out_both.split_at_mut(out_len);
    // Delayed restored output `(px0, pw_u)` from `unit_col - 1`, kept in
    // `out_prev` until `unit_col` has read the un-restored border `x0 - B .. x0`.
    let mut prev_unit: Option<(usize, usize)> = None;

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
            if let Some((px0, pw_u)) = prev_unit.take() {
                flush_prev(win, out_prev, y_first, h, px0, pw_u);
            }
            continue;
        }
        let w = x1 - x0;
        if x0 >= B && x1 + B <= pw {
            {
                let mut rows: [&[T]; LR_MAX_H + 2 * B] = [&[]; LR_MAX_H + 2 * B];
                let off = x0 - B;
                for (r, d) in rows[..B].iter_mut().enumerate() {
                    *d = &stripe_src_row(g, win, above, below, y_first, r)[off..];
                }
                let stride = win.stride;
                let base = (y_first - win.y0) * stride + off;
                let mut win_rows = &win.data()[base..];
                if let Some((last, head)) = rows[B..B + h].split_last_mut() {
                    for d in head {
                        *d = win_rows;
                        win_rows = &win_rows[stride..];
                    }
                    *last = win_rows;
                }
                for (r, d) in rows[..srows].iter_mut().enumerate().skip(B + h) {
                    *d = &stripe_src_row(g, win, above, below, y_first, r)[off..];
                }
                let dst = &mut out_cur[..w * h];
                match unit.typ {
                    RestorationType::Wiener => {
                        (dsp.wiener)(dst, w, &rows[..srows], w, h, &unit.wiener, bd)
                    }
                    _ => (dsp.sgr)(
                        dst,
                        w,
                        &rows[..srows],
                        w,
                        h,
                        unit.sgr_set as usize,
                        unit.sgr_xqd,
                        bd,
                    ),
                }
            }
            if let Some((px0, pw_u)) = prev_unit {
                flush_prev(win, out_prev, y_first, h, px0, pw_u);
            }
            std::mem::swap(&mut out_prev, &mut out_cur);
            prev_unit = Some((x0, w));
        } else {
            let sstride = w + 2 * B;
            for r in 0..srows {
                let row = stripe_src_row(g, win, above, below, y_first, r);
                copy_padded_row(
                    &mut edge_scratch[r * sstride..(r + 1) * sstride],
                    row,
                    x0,
                    x1,
                    pw,
                );
            }
            if let Some((px0, pw_u)) = prev_unit.take() {
                flush_prev(win, out_prev, y_first, h, px0, pw_u);
            }
            let mut rows: [&[T]; LR_MAX_H + 2 * B] = [&[]; LR_MAX_H + 2 * B];
            for (r, d) in rows[..srows].iter_mut().enumerate() {
                *d = &edge_scratch[r * sstride..(r + 1) * sstride + LR_SRC_SLACK];
            }
            let next_filtered = unit_col + 1 < lr.unit_cols
                && x1 < pw
                && matches!(
                    lr.units[unit_row * lr.unit_cols + unit_col + 1].typ,
                    RestorationType::Wiener | RestorationType::Sgrproj
                );
            if next_filtered {
                let dst = &mut out_cur[..w * h];
                match unit.typ {
                    RestorationType::Wiener => {
                        (dsp.wiener)(dst, w, &rows[..srows], w, h, &unit.wiener, bd)
                    }
                    _ => (dsp.sgr)(
                        dst,
                        w,
                        &rows[..srows],
                        w,
                        h,
                        unit.sgr_set as usize,
                        unit.sgr_xqd,
                        bd,
                    ),
                }
                std::mem::swap(&mut out_prev, &mut out_cur);
                prev_unit = Some((x0, w));
            } else {
                let stride = win.stride;
                let dst = win.slice_mut(x0, y_first);
                match unit.typ {
                    RestorationType::Wiener => {
                        (dsp.wiener)(dst, stride, &rows[..srows], w, h, &unit.wiener, bd)
                    }
                    _ => (dsp.sgr)(
                        dst,
                        stride,
                        &rows[..srows],
                        w,
                        h,
                        unit.sgr_set as usize,
                        unit.sgr_xqd,
                        bd,
                    ),
                }
            }
        }
    }
    if let Some((px0, pw_u)) = prev_unit {
        flush_prev(win, out_prev, y_first, h, px0, pw_u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::decode::state::{count_units_in_frame, LrUnit};
    use crate::decoder::av1::dsp::Dsp;

    /// `lr_stripe_in_place` matches `lr_stripe`, also when the last unit
    /// of the row is wider than `unit_size` (up to about 1.5x).
    #[test]
    fn in_place_matches_lr_stripe_with_wide_last_unit() {
        let dsp = Dsp::<u8>::new(8);
        let unit_size = 64;
        let plane_h = 40;
        let stride = 256;
        for pw in [90, 2 * 64 + 31, 3 * 64 + 31] {
            for typ in [RestorationType::Wiener, RestorationType::Sgrproj] {
                let unit_cols = count_units_in_frame(unit_size, pw);
                let unit = LrUnit {
                    typ,
                    wiener: [[3, -7, 15]; 2],
                    sgr_set: 0,
                    sgr_xqd: [-32, 31],
                };
                let lr = LrPlane {
                    unit_size,
                    unit_rows: 1,
                    unit_cols,
                    units: vec![unit; unit_cols],
                };
                let mut win = PlaneWin::<u8>::new(stride, plane_h);
                for (i, v) in win.data_mut().iter_mut().enumerate() {
                    *v = ((i * 37 + (i / stride) * 11) % 251) as u8;
                }
                let g = StripeGeom {
                    start_y: -8,
                    end_y: 55,
                    plane_w: pw,
                    plane_h,
                };
                let mut out = vec![0u8; stride * plane_h];
                let mut scratch = Vec::new();
                lr_stripe(
                    &lr,
                    &g,
                    &win,
                    None,
                    None,
                    &mut out,
                    stride,
                    &mut scratch,
                    0,
                    8,
                    (0, usize::MAX),
                    &dsp,
                );
                lr_stripe_in_place(&lr, &g, &mut win, None, None, &mut scratch, 0, 8, &dsp);
                for y in 0..plane_h {
                    assert_eq!(
                        &win.row(y)[..pw],
                        &out[y * stride..y * stride + pw],
                        "pw {pw} row {y}"
                    );
                }
            }
        }
    }
}
