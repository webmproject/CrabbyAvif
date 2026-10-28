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

//! Frame decoding pipeline: superblock rows are decoded (all tile columns
//! interleaved), then post-filtered with a small lag and handed to an
//! output sink in horizontal bands.

pub mod block;
pub mod buffers;
pub mod cdef_apply;
pub mod deblock;
pub mod intrabc;
pub mod lr_apply;
pub mod recon;
pub mod state;
pub mod superres;

use std::collections::VecDeque;

use crate::decoder::av1::error::{Error, Result};
use crate::decoder::av1::headers::RestorationType;
use crate::decoder::av1::obu::FrameData;
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::types::MI_SIZE;

use block::{SbDecoder, Scratch};
use buffers::{MiWin, PlaneWin};
use cdef_apply::CdefSrc;
use lr_apply::{LrBoundary, StripeGeom};
use state::{AboveCtx, FrameCtx, FrameState, IntraSetup, LeftCtx, LrPlane, TileState};
use superres::Upscaler;

/// Luma rows at the bottom of a deblocked superblock row that are
/// post-filtered (CDEF, superres, LR) only after the next row: deblocking
/// the next row's top edge changes up to 6 of them (13-tap luma filter;
/// chroma filters change at most 2 rows), and CDEF reads 2 rows below its
/// blocks. A multiple of 8 so that CDEF stays aligned to 8x8 blocks.
const POST_LAG: usize = 8;

/// Rows of one plane handed to the sink.
pub struct PlaneRows<'a, T> {
    /// First plane row of the band.
    pub y0: usize,
    /// Number of rows.
    pub h: usize,
    /// Plane width (samples per row that are valid).
    pub w: usize,
    pub data: &'a [T],
    pub stride: usize,
}

/// Receives finished output rows, top to bottom. Every call covers the
/// rows following the previous call (for each plane).
pub trait RowSink<T: Pixel> {
    fn push(&mut self, planes: &[PlaneRows<'_, T>]) -> Result<()>;

    /// Whether the sink can take the whole decoded frame at once
    /// ([`RowSink::finish`]) instead of rows. The decoder then decodes
    /// and filters directly into full frame planes that become the
    /// output (no copies), unless it decodes a region or uses super
    /// resolution.
    fn wants_frame(&self) -> bool {
        false
    }

    /// Receives the finished frame (one plane per color plane, possibly
    /// with `stride > width`); only called if [`RowSink::wants_frame`]
    /// and then instead of any [`RowSink::push`].
    fn finish(&mut self, planes: Vec<crate::decoder::av1::sink::Plane<T>>) -> Result<()> {
        let _ = planes;
        Ok(())
    }

    /// Only in [`RowSink::wants_frame`] mode: luma rows `0..y_end` (and
    /// the matching chroma rows) of the frame planes `planes` (row 0 at
    /// index 0, `strides` per plane) are final and the decoder never
    /// reads them again, so the sink may modify them in place (film
    /// grain while they are still in cache). Called top to bottom, with
    /// `y_end == frame_height` last.
    fn rows_final(&mut self, planes: &mut [&mut [T]], strides: &[usize], y_end: usize) {
        let _ = (planes, strides, y_end);
    }
}

/// Which in-loop filters to run (for testing; all on by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterMask {
    pub deblock: bool,
    pub cdef: bool,
    pub restoration: bool,
}

impl Default for FilterMask {
    fn default() -> Self {
        Self {
            deblock: true,
            cdef: true,
            restoration: true,
        }
    }
}

/// Region of superblocks that has to be decoded.
#[derive(Clone, Copy, Debug)]
pub struct SbRegion {
    /// Superblock rows `0..row_end` are needed (rows below are skipped).
    pub row_end: usize,
    /// Superblock columns `col_start..col_end` intersect the output region
    /// (grown by the filter margin).
    pub col_start: usize,
    pub col_end: usize,
    /// Superblock rows `row_start..row_end` intersect the output region.
    pub row_start: usize,
}

pub struct DecodeParams {
    pub filters: FilterMask,
    pub region: Option<SbRegion>,
    pub threads: usize,
}

struct Pipeline<'a, T: Pixel> {
    f: FrameCtx<'a>,
    s: FrameState<T>,
    sc: Scratch,
    deblock: bool,
    cdef: bool,
    lr: bool,
    superres: bool,
    /// Plane sizes of the output (upscaled) frame.
    out_w: [usize; 3],
    out_h: [usize; 3],
    /// Plane widths / heights of the coded (mi aligned) frame.
    mi_w: [usize; 3],
    mi_h: [usize; 3],
    /// Pre-CDEF rows above the current superblock row (`cdef_line[cur]`)
    /// and the bottom rows of the current one for the next row.
    cdef_line: [Vec<[Vec<T>; 2]>; 2],
    cdef_cur: usize,
    cdef_srcs: Vec<CdefSrc>,
    up: Vec<PlaneWin<T>>,
    upscalers: Vec<Upscaler>,
    lr_bounds: VecDeque<LrBoundary<T>>,
    lr_free: Vec<LrBoundary<T>>,
    next_stripe: usize,
    /// Next LR stripe boundary whose deblocked rows are to be saved.
    next_bound: usize,
    lr_out: Vec<Vec<T>>,
    lr_scratch: Vec<T>,
    /// Deblocking edge parameter scratch.
    lf_grid: Vec<u8>,
    intrabc: bool,
    /// Decoding into full frame planes that are handed to the sink at
    /// the end (see `RowSink::wants_frame`).
    direct: bool,
    /// Pre loop restoration samples of the last columns of the previous
    /// restoration unit (in-place loop restoration).
    lr_left: Vec<T>,
    /// DSP kernel table.
    dsp: crate::decoder::av1::dsp::Dsp<T>,
    /// First superblock row that is post-filtered and output.
    post_start: usize,
    /// Mode-info column range that needs post-filtering.
    col_range: (usize, usize),
    /// Luma rows `0..rows_end` are decoded (the whole frame unless a region
    /// is decoded).
    rows_end: usize,
}

pub fn decode_frame<T: Pixel>(
    fd: &FrameData,
    params: &DecodeParams,
    sink: &mut dyn RowSink<T>,
) -> Result<()> {
    let f = FrameCtx::new(&fd.seq, &fd.frame);
    let fh = f.fh;
    let np = f.num_planes;
    let sb_px = f.sb_px;
    let intrabc = fh.allow_intrabc;
    let mi_cols_al = f.sb_cols * f.sb4;
    let mi_rows_al = f.sb_rows * f.sb4;
    let mut planes = Vec::new();
    let mut above_line = Vec::new();
    let mut mi_w = [0; 3];
    let mut mi_h = [0; 3];
    let mut out_w = [0; 3];
    let mut out_h = [0; 3];
    // Without intra block copy only rows from the superblock row above
    // (minus 16) on are needed (2 * sb_px + 16 rows); the spare rows
    // make the window move (copy) only every third superblock row.
    let direct = params.region.is_none() && !fh.use_superres && sink.wants_frame();
    let win_rows_luma = if intrabc || direct { mi_rows_al * 4 } else { 4 * sb_px + 16 };
    for p in 0..np {
        let (sx, sy) = f.sub(p);
        mi_w[p] = (f.mi_cols * 4) >> sx;
        mi_h[p] = (f.mi_rows * 4) >> sy;
        out_w[p] = (fh.upscaled_width + sx) >> sx;
        out_h[p] = (fh.frame_height + sy) >> sy;
        // A multiple of 64 bytes, so that every row starts 64-byte aligned
        // (PlaneWin aligns the first one).
        let stride = (((mi_cols_al * 4) >> sx) + 64).next_multiple_of(64);
        planes.push(PlaneWin::new(stride, win_rows_luma >> sy));
        above_line.push(vec![T::default(); stride]);
    }
    let above_w = mi_cols_al + 32;
    let deblock =
        params.filters.deblock && !intrabc && (fh.lf.level[0] != 0 || fh.lf.level[1] != 0);
    let deblock_uv = deblock && np > 1 && (fh.lf.level[2] != 0 || fh.lf.level[3] != 0);
    let s = FrameState {
        planes,
        above_line,
        above_line_y: [0; 3],
        mi: MiWin::new(
            mi_cols_al + 32,
            2 * f.sb4,
            f.ss_x,
            f.ss_y,
            deblock,
            deblock_uv,
            deblock && fh.delta_lf_present,
            intrabc,
        ),
        above: AboveCtx {
            col0: 0,
            level: [vec![0; above_w], vec![0; above_w], vec![0; above_w]],
            dc: [vec![0; above_w], vec![0; above_w], vec![0; above_w]],
            palette: [vec![[0; 8]; above_w], vec![[0; 8]; above_w]],
        },
        left: LeftCtx::new(),
        cdef_idx: vec![-1; (mi_rows_al / 16 + 2) * (mi_cols_al / 16 + 2)],
        cdef_stride: mi_cols_al / 16 + 2,
        cdef_row0: 0,
        cdef_col0: 0,
        lr: [
            LrPlane::new(fh, 0, f.ss_x, f.ss_y),
            LrPlane::new(fh, 1, f.ss_x, f.ss_y),
            LrPlane::new(fh, 2, f.ss_x, f.ss_y),
        ],
        cfl_ac: Box::new([0; crate::decoder::av1::dsp::ipred::MAX_CFL_AC]),
        edges: crate::decoder::av1::dsp::ipred::IntraEdges::default(),
        ipred: [IntraSetup::new(&f, 0), IntraSetup::new(&f, 1)],
    };
    let superres = fh.use_superres;
    let lr = params.filters.restoration && fh.lr.uses_lr;
    let cdef = params.filters.cdef && !intrabc && f.seq.enable_cdef && !fh.coded_lossless;
    let mut p = Pipeline {
        deblock,
        cdef,
        lr,
        superres,
        out_w,
        out_h,
        mi_w,
        mi_h,
        cdef_line: if cdef {
            [0, 1].map(|_| {
                (0..np)
                    .map(|p| [vec![T::default(); mi_w[p]], vec![T::default(); mi_w[p]]])
                    .collect()
            })
        } else {
            [Vec::new(), Vec::new()]
        },
        cdef_cur: 0,
        cdef_srcs: (0..np).map(|_| CdefSrc::new()).collect(),
        up: Vec::new(),
        upscalers: Vec::new(),
        lr_bounds: VecDeque::new(),
        lr_free: Vec::new(),
        next_stripe: 0,
        next_bound: 1,
        lr_out: (0..np)
            .map(|p| {
                if lr && !direct && fh.lr.frame_restoration_type[p] != RestorationType::None {
                    vec![T::default(); out_w[p] * (64 >> f.sub(p).1)]
                } else {
                    Vec::new()
                }
            })
            .collect(),
        lr_scratch: Vec::new(),
        lf_grid: Vec::new(),
        intrabc,
        direct,
        lr_left: Vec::new(),
        dsp: crate::decoder::av1::dsp::Dsp::new(f.bd),
        post_start: 0,
        col_range: (0, usize::MAX),
        rows_end: usize::MAX,
        s,
        sc: Scratch::new(),
        f,
    };
    if superres {
        for pl in 0..np {
            let (sx, sy) = p.f.sub(pl);
            p.upscalers.push(Upscaler::new(
                fh.frame_width,
                fh.upscaled_width,
                p.f.mi_cols,
                sx,
            ));
            p.up.push(PlaneWin::new(p.out_w[pl] + 16, win_rows_luma >> sy));
        }
    }
    p.run(fd, params, sink)
}

impl<'a, T: Pixel> Pipeline<'a, T> {
    fn run(
        &mut self,
        fd: &FrameData,
        params: &DecodeParams,
        sink: &mut dyn RowSink<T>,
    ) -> Result<()> {
        let f = &self.f;
        let ti = &f.fh.tile_info;
        let sb4 = f.sb4;
        let sb_rows = f.sb_rows;
        let fh = f.fh;
        let full = SbRegion {
            row_start: 0,
            row_end: sb_rows,
            col_start: 0,
            col_end: f.sb_cols,
        };
        let reg = params.region.unwrap_or(full);
        let row_end = reg.row_end.min(sb_rows);
        let col_end = reg.col_end.min(f.sb_cols);
        // Tiles are independent: only tile rows / columns that intersect the
        // region are decoded at all.
        let tile_sb = |mi: usize| mi / sb4;
        let tile_rows_needed: Vec<bool> = (0..ti.rows)
            .map(|t| {
                tile_sb(ti.mi_row_starts[t]) < row_end
                    && tile_sb(ti.mi_row_starts[t + 1] + sb4 - 1) > reg.row_start
            })
            .collect();
        let tile_cols_needed: Vec<bool> = (0..ti.cols)
            .map(|t| {
                tile_sb(ti.mi_col_starts[t]) < col_end
                    && tile_sb(ti.mi_col_starts[t + 1] + sb4 - 1) > reg.col_start
            })
            .collect();
        // needed[tr][tc]: tile has to be parsed. recon_tile[tr][tc]: tile
        // intersects the region and is reconstructed.
        let mut needed = vec![vec![false; ti.cols]; ti.rows];
        for tr in 0..ti.rows {
            for tc in 0..ti.cols {
                needed[tr][tc] = tile_rows_needed[tr] && tile_cols_needed[tc];
            }
        }
        let recon_tile = needed.clone();
        if params.region.is_some() && fh.lr.uses_lr && !self.intrabc {
            // Loop restoration coefficients of a unit are coded in the
            // superblock containing the unit's top-left corner, which may lie
            // in a tile outside the region: those tiles are parsed too.
            for (tr, tc) in self.lr_coding_tiles(&reg, row_end, col_end) {
                needed[tr][tc] = true;
            }
        }
        let first_tile_row = needed
            .iter()
            .position(|r| r.iter().any(|&n| n))
            .unwrap_or(0);
        let first_row = tile_sb(ti.mi_row_starts[first_tile_row]);
        // Rows above the region are reconstructed (intra prediction depends
        // on them) but not post-filtered or output.
        self.post_start = reg.row_start.max(tile_sb(
            ti.mi_row_starts[tile_rows_needed.iter().position(|&n| n).unwrap_or(0)],
        ));
        self.col_range = (reg.col_start * sb4, col_end * sb4);
        self.next_stripe = (self.post_start * f.sb_px + 8) / 64;
        // Boundary k lies at luma row 64k - 8.
        self.next_bound = (self.post_start * f.sb_px + 8).div_ceil(64).max(1);
        self.rows_end = (row_end * f.sb_px).min(fh.frame_height);
        let mut tile_jobs = Vec::new();
        for tr in 0..ti.rows {
            for tc in 0..ti.cols {
                if needed[tr][tc] {
                    tile_jobs.push((tr, tc));
                }
            }
        }
        if params.threads > 1 && tile_jobs.len() > 1 {
            return self.run_mt(
                fd,
                params,
                sink,
                &tile_jobs,
                &recon_tile,
                first_row,
                row_end,
                col_end,
            );
        }
        let mut tiles: Vec<Option<TileState>> = Vec::new();
        let mut cur_tile_row = usize::MAX;
        for r in first_row..row_end {
            let mi_row = r * sb4;
            // Tile row containing this superblock row.
            let tr = (0..ti.rows)
                .find(|&t| mi_row >= ti.mi_row_starts[t] && mi_row < ti.mi_row_starts[t + 1])
                .unwrap();
            if tr != cur_tile_row {
                cur_tile_row = tr;
                tiles.clear();
                for tc in 0..ti.cols {
                    if !needed[tr][tc] {
                        tiles.push(None);
                        continue;
                    }
                    let data = fd.tiles[tr * ti.cols + tc];
                    let ts = TileState::new(
                        data,
                        fh,
                        (ti.mi_row_starts[tr], ti.mi_row_starts[tr + 1]),
                        (ti.mi_col_starts[tc], ti.mi_col_starts[tc + 1]),
                    );
                    tiles.push(Some(ts));
                }
                // clear_above_context()
                for a in self
                    .s
                    .above
                    .level
                    .iter_mut()
                    .chain(self.s.above.dc.iter_mut())
                {
                    a.fill(0);
                }
            }
            self.slide_windows(r);
            // Decode the superblock row of every needed tile column.
            let sb_y = r * self.f.sb_px;
            for (tc, ts) in tiles.iter_mut().enumerate() {
                let Some(ts) = ts else { continue };
                let recon_this_tile = recon_tile[tr][tc];
                self.s.left.clear();
                let c0 = ts.mi_col_start;
                let c1 = ts.mi_col_end;
                // Intra dependency cone: a superblock depends on the one
                // above-right of it, so one more superblock column is needed
                // for every superblock row above the bottom of the region.
                // Intra block copy may reference anything in the tile.
                let cone_end = if self.intrabc || params.region.is_none() {
                    c1
                } else {
                    c1.min((col_end + (row_end - 1 - r)) * sb4)
                };
                let mut dec = SbDecoder {
                    f: &self.f,
                    t: ts,
                    s: &mut self.s,
                    sc: &mut self.sc,
                    dsp: &self.dsp,
                    b: Default::default(),
                    recon: true,
                    sb_y,
                    read_deltas: false,
                    plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
                };
                let mut c = c0;
                while c < c1 {
                    dec.recon = recon_this_tile && c < cone_end;
                    dec.decode_sb(mi_row, c)?;
                    if dec.t.msac.overrun() {
                        // More than the allowed 14 bits of zero padding
                        // consumed: the tile data is truncated.
                        return Err(Error::Truncated);
                    }
                    c += sb4;
                }
            }
            self.after_decode_row(r)?;
            if r >= self.post_start {
                self.post_row(r, r + 1 == row_end, sink)?;
            }
        }
        if self.direct {
            // Hand the planes over as the output frame.
            let planes = (0..self.f.num_planes)
                .map(|p| {
                    let w = std::mem::replace(&mut self.s.planes[p], PlaneWin::new(0, 0));
                    debug_assert!(
                        w.y0 == 0 && w.rows >= self.out_h[p] && w.stride >= self.out_w[p]
                    );
                    let stride = w.stride;
                    let (data, offset) = w.into_raw();
                    crate::decoder::av1::sink::Plane {
                        width: self.out_w[p],
                        height: self.out_h[p],
                        stride,
                        offset,
                        data,
                    }
                })
                .collect();
            sink.finish(planes)?;
        }
        Ok(())
    }

    /// Decoded luma rows including the coded padding below the frame.
    fn rows_end_mi_aligned(&self) -> usize {
        if self.rows_end >= self.f.fh.frame_height {
            self.f.mi_rows * 4
        } else {
            self.rows_end
        }
    }

    /// First luma row post-filtered by `post_row(r)` (see there).
    fn post_row_start(&self, r: usize) -> usize {
        if r <= self.post_start {
            r * self.f.sb_px
        } else {
            r * self.f.sb_px - POST_LAG
        }
    }

    /// Tiles (row, col) that code the loop restoration units overlapping the
    /// superblock region.
    fn lr_coding_tiles(
        &self,
        reg: &SbRegion,
        row_end: usize,
        col_end: usize,
    ) -> Vec<(usize, usize)> {
        let f = &self.f;
        let fh = f.fh;
        let ti = &fh.tile_info;
        let mut out = Vec::new();
        let tile_of = |starts: &[usize], mi: usize| -> usize {
            (0..starts.len() - 1)
                .find(|&t| mi >= starts[t] && mi < starts[t + 1])
                .unwrap_or(starts.len() - 2)
        };
        for plane in 0..f.num_planes {
            if fh.lr.frame_restoration_type[plane] == RestorationType::None {
                continue;
            }
            let lp = &self.s.lr[plane];
            let us = lp.unit_size;
            let (sx, sy) = f.sub(plane);
            let y0 = (reg.row_start * f.sb_px) >> sy;
            let y1 = ((row_end * f.sb_px) >> sy).max(y0 + 1);
            let unit_row = |y: usize| (lp.unit_rows - 1).min((y + (8 >> sy)) / us);
            // Upscaled plane columns of the region.
            let x0 = ((reg.col_start * f.sb_px * fh.upscaled_width / fh.frame_width.max(1)) >> sx)
                .min(self.out_w[plane] - 1);
            let x1 =
                ((col_end * f.sb_px * fh.upscaled_width / fh.frame_width.max(1)) >> sx).max(x0 + 1);
            let unit_col = |x: usize| (lp.unit_cols - 1).min(x / us);
            let (num, den) = if fh.use_superres {
                (
                    (MI_SIZE >> sx) * fh.superres_denom as usize,
                    us * crate::decoder::av1::types::SUPERRES_NUM as usize,
                )
            } else {
                (MI_SIZE >> sx, us)
            };
            for ur in unit_row(y0)..=unit_row(y1 - 1) {
                // read_lr reads unit row ur in the superblock row containing
                // plane row ur * us.
                let mi_row = (((ur * us) << sy) / MI_SIZE).min(f.mi_rows - 1);
                for uc in unit_col(x0)..=unit_col(x1 - 1) {
                    let mi_col = ((uc * den / num) / f.sb4 * f.sb4).min(f.mi_cols - 1);
                    out.push((
                        tile_of(&ti.mi_row_starts, mi_row),
                        tile_of(&ti.mi_col_starts, mi_col),
                    ));
                }
            }
        }
        out
    }

    fn slide_windows(&mut self, r: usize) {
        let f = &self.f;
        let m_y0 = (r * f.sb4).saturating_sub(f.sb4);
        self.s.mi.slide_to(m_y0);
        if self.intrabc || self.direct {
            return;
        }
        let y0 = (r * f.sb_px).saturating_sub(f.sb_px + 16);
        let need_end = (r + 1) * f.sb_px;
        let keep_end = r * f.sb_px;
        for (p, pl) in self
            .s
            .planes
            .iter_mut()
            .chain(self.up.iter_mut())
            .enumerate()
        {
            let sy = f.sub(p % f.num_planes).1;
            pl.slide_to(y0 >> sy, need_end >> sy, keep_end >> sy);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mt(
        &mut self,
        fd: &FrameData,
        params: &DecodeParams,
        sink: &mut dyn RowSink<T>,
        tile_jobs: &[(usize, usize)],
        recon_tile: &[Vec<bool>],
        first_row: usize,
        row_end: usize,
        col_end: usize,
    ) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let f = &self.f;
        let fh = f.fh;
        let ti = &fh.tile_info;
        let sb4 = f.sb4;
        let sb_px = f.sb_px;
        let np = f.num_planes;
        let deblock = self.deblock;
        let deblock_uv = deblock && np > 1 && (fh.lf.level[2] != 0 || fh.lf.level[3] != 0);
        let dsp = &self.dsp;
        let post_start = self.post_start;
        let has_region = params.region.is_some();

        let num_jobs = tile_jobs.len();
        let num_workers = params.threads.min(num_jobs);
        let next_job = AtomicUsize::new(0);
        let min_failed_key = AtomicUsize::new(usize::MAX);

        let worker_results: Vec<
            Vec<(
                usize,
                usize,
                std::result::Result<FrameState<T>, (usize, Error)>,
            )>,
        > = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_workers.saturating_sub(1));
            for _ in 1..num_workers {
                handles.push(s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let idx = next_job.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_jobs {
                            break;
                        }
                        let (tr, tc) = tile_jobs[idx];
                        let data = fd.tiles[tr * ti.cols + tc];
                        let res = Self::decode_tile_unit(
                            f,
                            dsp,
                            data,
                            tr,
                            tc,
                            recon_tile[tr][tc],
                            deblock,
                            deblock_uv,
                            row_end,
                            col_end,
                            post_start,
                            has_region,
                            &min_failed_key,
                        );
                        if let Err((key, _)) = &res {
                            min_failed_key.fetch_min(*key, Ordering::Relaxed);
                        }
                        out.push((tr, tc, res));
                    }
                    out
                }));
            }
            let mut main_out = Vec::new();
            loop {
                let idx = next_job.fetch_add(1, Ordering::Relaxed);
                if idx >= num_jobs {
                    break;
                }
                let (tr, tc) = tile_jobs[idx];
                let data = fd.tiles[tr * ti.cols + tc];
                let res = Self::decode_tile_unit(
                    f,
                    dsp,
                    data,
                    tr,
                    tc,
                    recon_tile[tr][tc],
                    deblock,
                    deblock_uv,
                    row_end,
                    col_end,
                    post_start,
                    has_region,
                    &min_failed_key,
                );
                if let Err((key, _)) = &res {
                    min_failed_key.fetch_min(*key, Ordering::Relaxed);
                }
                main_out.push((tr, tc, res));
            }
            let mut all = Vec::with_capacity(num_workers);
            all.push(main_out);
            for h in handles {
                all.push(h.join().unwrap());
            }
            all
        });

        let mut tile_grid: Vec<Vec<Option<FrameState<T>>>> = (0..ti.rows)
            .map(|_| (0..ti.cols).map(|_| None).collect())
            .collect();
        let mut first_err: Option<(usize, Error)> = None;
        for batch in worker_results {
            for (tr, tc, res) in batch {
                match res {
                    Ok(tile_s) => tile_grid[tr][tc] = Some(tile_s),
                    Err((key, e)) => {
                        if first_err.as_ref().is_none_or(|(k, _)| key < *k) {
                            first_err = Some((key, e));
                        }
                    }
                }
            }
        }
        if let Some((_, e)) = first_err {
            return Err(e);
        }

        // Merge loop restoration units and cdef_idx into self.s.
        for tr in 0..ti.rows {
            let mi_r0 = ti.mi_row_starts[tr];
            let mi_r1 = ti.mi_row_starts[tr + 1];
            let sb_r0 = mi_r0 / sb4;
            let sb_r1 = mi_r1.div_ceil(sb4).min(row_end);
            for tc in 0..ti.cols {
                let Some(ts) = &tile_grid[tr][tc] else {
                    continue;
                };
                let mi_c0 = ti.mi_col_starts[tc];
                let mi_c1 = ti.mi_col_starts[tc + 1];
                let sb_c1 = mi_c1.div_ceil(sb4);
                if fh.lr.uses_lr && !self.intrabc {
                    for plane in 0..np {
                        if fh.lr.frame_restoration_type[plane] == RestorationType::None {
                            continue;
                        }
                        let (sx, sy) = f.sub(plane);
                        let lp = &mut self.s.lr[plane];
                        let t_lp = &ts.lr[plane];
                        let unit_size = lp.unit_size;
                        let unit_rows = lp.unit_rows;
                        let unit_cols = lp.unit_cols;
                        let unit_row_start = ((sb_r0 * sb4) * (MI_SIZE >> sy)).div_ceil(unit_size);
                        let unit_row_end =
                            unit_rows.min(((sb_r1 * sb4) * (MI_SIZE >> sy)).div_ceil(unit_size));
                        let (numerator, denominator) = if fh.use_superres {
                            (
                                (MI_SIZE >> sx) * fh.superres_denom as usize,
                                unit_size * crate::decoder::av1::types::SUPERRES_NUM as usize,
                            )
                        } else {
                            (MI_SIZE >> sx, unit_size)
                        };
                        let unit_col_start = (mi_c0 * numerator).div_ceil(denominator);
                        let unit_col_end =
                            unit_cols.min(((sb_c1 * sb4) * numerator).div_ceil(denominator));
                        for ur in unit_row_start..unit_row_end {
                            let base = ur * unit_cols;
                            if unit_col_start < unit_col_end {
                                lp.units[base + unit_col_start..base + unit_col_end]
                                    .copy_from_slice(
                                        &t_lp.units[base + unit_col_start..base + unit_col_end],
                                    );
                            }
                        }
                    }
                }
                if recon_tile[tr][tc] && self.cdef {
                    let cdef_r0 = sb_r0 * sb4 / 16;
                    let cdef_r1 = (sb_r1 * sb4).div_ceil(16);
                    let cdef_c0 = mi_c0 / 16;
                    let cdef_c1 = mi_c1.div_ceil(16);
                    let dst_stride = self.s.cdef_stride;
                    let src_stride = ts.cdef_stride;
                    for cr in cdef_r0..cdef_r1 {
                        let dst_i = cr * dst_stride + cdef_c0;
                        let src_i = (cr - ts.cdef_row0) * src_stride + (cdef_c0 - ts.cdef_col0);
                        let len = cdef_c1 - cdef_c0;
                        self.s.cdef_idx[dst_i..dst_i + len]
                            .copy_from_slice(&ts.cdef_idx[src_i..src_i + len]);
                    }
                }
            }
        }

        // Run row-by-row post-filtering and output sink.
        let copy_start = if self.intrabc {
            first_row
        } else {
            first_row.max(self.post_start.saturating_sub(1))
        };
        for r in copy_start..row_end {
            let mi_row = r * sb4;
            let tr = (0..self.f.fh.tile_info.rows)
                .find(|&t| {
                    mi_row >= self.f.fh.tile_info.mi_row_starts[t]
                        && mi_row < self.f.fh.tile_info.mi_row_starts[t + 1]
                })
                .unwrap();
            self.slide_windows(r);
            let f = &self.f;
            let ti = &f.fh.tile_info;
            let sb_y = r * sb_px;
            for tc in 0..ti.cols {
                if !recon_tile[tr][tc] {
                    continue;
                }
                let Some(ts) = &tile_grid[tr][tc] else {
                    continue;
                };
                let mi_c0 = ti.mi_col_starts[tc];
                let mi_c1 = ti.mi_col_starts[tc + 1];
                let cone_end = if self.intrabc || params.region.is_none() {
                    mi_c1
                } else {
                    mi_c1.min((col_end + (row_end - 1 - r)) * sb4)
                };
                if mi_c0 >= cone_end {
                    continue;
                }
                // Copy planes
                for p in 0..np {
                    let (sx, sy) = f.sub(p);
                    let x0 = (mi_c0 * 4) >> sx;
                    let x1 = ((cone_end * 4) >> sx).min(self.mi_w[p]);
                    if x0 >= x1 {
                        continue;
                    }
                    let y0 = sb_y >> sy;
                    let y1 = ((sb_y + sb_px) >> sy).min(self.mi_h[p]);
                    let len = x1 - x0;
                    for y in y0..y1 {
                        let dst_idx = self.s.planes[p].idx(x0, y);
                        let src_idx = ts.planes[p].idx(x0, y);
                        self.s.planes[p].data_mut()[dst_idx..dst_idx + len]
                            .copy_from_slice(&ts.planes[p].data()[src_idx..src_idx + len]);
                    }
                }
                // Copy mi info (needed for deblock and CDEF)
                if self.deblock || self.cdef {
                    let mi_y0 = mi_row;
                    let mi_y1 = (mi_row + sb4).min(f.mi_rows);
                    let len = cone_end - mi_c0;
                    for y in mi_y0..mi_y1 {
                        let dst_idx = self.s.mi.idx(y, mi_c0);
                        let src_idx = ts.mi.idx(y, mi_c0);
                        self.s.mi.data[dst_idx..dst_idx + len]
                            .copy_from_slice(&ts.mi.data[src_idx..src_idx + len]);
                        if !self.s.mi.delta_lf.is_empty() {
                            self.s.mi.delta_lf[dst_idx..dst_idx + len]
                                .copy_from_slice(&ts.mi.delta_lf[src_idx..src_idx + len]);
                        }
                    }
                    for pt in 0..2 {
                        if self.s.mi.lf_tx[pt].is_empty() {
                            continue;
                        }
                        let sx = if pt == 0 { 0 } else { f.ss_x };
                        let sy = if pt == 0 { 0 } else { f.ss_y };
                        let lf_x0 = mi_c0 >> sx;
                        let lf_x1 = cone_end >> sx;
                        let lf_y0 = mi_y0 >> sy;
                        let lf_y1 = (mi_y1 + sy) >> sy;
                        let len = lf_x1 - lf_x0;
                        let uv_col0 = ts.mi.col0 >> sx;
                        for y in lf_y0..lf_y1 {
                            let dst_row = self.s.mi.lf_tx_row_mut(pt, y);
                            let src_row = ts.mi.lf_tx_row(pt, y);
                            dst_row[lf_x0..lf_x0 + len]
                                .copy_from_slice(&src_row[lf_x0 - uv_col0..lf_x0 - uv_col0 + len]);
                        }
                    }
                }
            }
            if self.deblock && r >= self.post_start {
                deblock::deblock_rows(
                    f,
                    &self.s.mi,
                    &mut self.s.planes,
                    r * sb4,
                    (r + 1) * sb4,
                    self.col_range,
                    &self.dsp,
                    &mut self.lf_grid,
                );
            }
            if r >= self.post_start {
                self.post_row(r, r + 1 == row_end, sink)?;
            }
        }
        if self.direct {
            let planes = (0..self.f.num_planes)
                .map(|p| {
                    let w = std::mem::replace(&mut self.s.planes[p], PlaneWin::new(0, 0));
                    debug_assert!(
                        w.y0 == 0 && w.rows >= self.out_h[p] && w.stride >= self.out_w[p]
                    );
                    let stride = w.stride;
                    let (data, offset) = w.into_raw();
                    crate::decoder::av1::sink::Plane {
                        width: self.out_w[p],
                        height: self.out_h[p],
                        stride,
                        offset,
                        data,
                    }
                })
                .collect();
            sink.finish(planes)?;
        }
        Ok(())
    }

    /// Decodes a single tile `(tr, tc)` as an independent AV1 unit into its own compact `FrameState<T>`.
    #[allow(clippy::too_many_arguments)]
    fn decode_tile_unit(
        f: &FrameCtx<'a>,
        dsp: &crate::decoder::av1::dsp::Dsp<T>,
        data: &[u8],
        tr: usize,
        tc: usize,
        recon: bool,
        deblock: bool,
        deblock_uv: bool,
        row_end: usize,
        col_end: usize,
        post_start: usize,
        has_region: bool,
        min_failed_key: &std::sync::atomic::AtomicUsize,
    ) -> std::result::Result<FrameState<T>, (usize, Error)> {
        let ti = &f.fh.tile_info;
        let fh = f.fh;
        let sb4 = f.sb4;
        let sb_px = f.sb_px;
        let intrabc = fh.allow_intrabc;
        let mi_r0 = ti.mi_row_starts[tr];
        let mi_r1 = ti.mi_row_starts[tr + 1];
        let mi_c0 = ti.mi_col_starts[tc];
        let mi_c1 = ti.mi_col_starts[tc + 1];

        let sb_r0 = mi_r0 / sb4;
        let sb_r1 = mi_r1.div_ceil(sb4).min(row_end);
        // Check if an earlier tile row / superblock row has already failed before allocating.
        let start_key = (sb_r0 << 16) | (tr << 8) | tc;
        if start_key > min_failed_key.load(std::sync::atomic::Ordering::Relaxed) {
            return Err((usize::MAX, Error::Truncated));
        }
        let sb_cols_tile = (mi_c1 - mi_c0).div_ceil(sb4);
        let mi_cols_tile_al = sb_cols_tile * sb4;

        let retain_r0 = if intrabc { sb_r0 } else { sb_r0.max(post_start.saturating_sub(1)) };
        let retain_rows = sb_r1.saturating_sub(retain_r0);

        let np = f.num_planes;
        let mut planes = Vec::with_capacity(np);
        let win_rows_luma = if !recon { 0 } else { (retain_rows + 2) * sb_px + 64 };
        for p in 0..np {
            let (sx, sy) = f.sub(p);
            let x0 = (mi_c0 * 4) >> sx;
            let y0 = (sb_r0 * sb_px) >> sy;
            let stride = if recon { ((mi_cols_tile_al * 4) >> sx) + 64 } else { 0 };
            let rows = if recon { (win_rows_luma >> sy) + 16 } else { 0 };
            planes.push(PlaneWin::new_tile(stride, rows, x0, y0));
        }

        let above_w = mi_cols_tile_al + 32;
        let mi_rows = if recon { ((retain_rows + 2) * sb4).next_power_of_two() } else { 2 * sb4 };
        let cdef_row0 = (sb_r0 * sb4) / 16;
        let cdef_col0 = mi_c0 / 16;
        let cdef_rows = (sb_r1.saturating_sub(sb_r0) * sb4) / 16 + 2;
        let cdef_stride = mi_cols_tile_al / 16 + 2;

        let mut tile_s = FrameState {
            planes,
            // No deblocking runs while decoding the tile, so `planes[p]` holds
            // unfiltered rows directly and `above_line` is not needed.
            above_line: vec![Vec::new(); np],
            above_line_y: [0; 3],
            mi: MiWin::new_tile(
                mi_cols_tile_al + 32,
                mi_rows,
                mi_c0,
                (sb_r0 * sb4).saturating_sub(sb4),
                f.ss_x,
                f.ss_y,
                deblock && recon,
                deblock_uv && recon,
                deblock && fh.delta_lf_present && recon,
                intrabc,
            ),
            above: AboveCtx {
                col0: mi_c0,
                level: [vec![0; above_w], vec![0; above_w], vec![0; above_w]],
                dc: [vec![0; above_w], vec![0; above_w], vec![0; above_w]],
                palette: [vec![[0; 8]; above_w], vec![[0; 8]; above_w]],
            },
            left: LeftCtx::new(),
            cdef_idx: vec![-1; cdef_rows * cdef_stride],
            cdef_stride,
            cdef_row0,
            cdef_col0,
            lr: [
                LrPlane::new(fh, 0, f.ss_x, f.ss_y),
                LrPlane::new(fh, 1, f.ss_x, f.ss_y),
                LrPlane::new(fh, 2, f.ss_x, f.ss_y),
            ],
            cfl_ac: Box::new([0; crate::decoder::av1::dsp::ipred::MAX_CFL_AC]),
            edges: crate::decoder::av1::dsp::ipred::IntraEdges::default(),
            ipred: [IntraSetup::new(f, 0), IntraSetup::new(f, 1)],
        };

        let mut ts = TileState::new(data, fh, (mi_r0, mi_r1), (mi_c0, mi_c1));
        let mut sc = Scratch::new();

        for r in sb_r0..sb_r1 {
            let row_key = (r << 16) | (tr << 8) | tc;
            if row_key > min_failed_key.load(std::sync::atomic::Ordering::Relaxed) {
                return Err((usize::MAX, Error::Truncated));
            }
            let mi_row = r * sb4;
            let sb_y = r * sb_px;
            let m_y0 = if recon {
                (r * sb4).saturating_sub(sb4).min(retain_r0 * sb4)
            } else {
                (r * sb4).saturating_sub(sb4)
            };
            tile_s.mi.slide_to(m_y0);
            if recon && !intrabc {
                let min_y0 = (r * sb_px)
                    .saturating_sub(sb_px + 16)
                    .min(retain_r0 * sb_px);
                let need_end = (r + 1) * sb_px;
                let keep_end = r * sb_px;
                for p in 0..np {
                    let sy = f.sub(p).1;
                    tile_s.planes[p].slide_to(min_y0 >> sy, need_end >> sy, keep_end >> sy);
                }
            }
            tile_s.left.clear();
            let cone_end = if intrabc || !has_region {
                mi_c1
            } else {
                mi_c1.min((col_end + (row_end - 1 - r)) * sb4)
            };
            let mut dec = SbDecoder {
                f,
                t: &mut ts,
                s: &mut tile_s,
                sc: &mut sc,
                dsp,
                b: Default::default(),
                recon: true,
                sb_y,
                read_deltas: false,
                plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
            };
            let mut c = mi_c0;
            while c < mi_c1 {
                dec.recon = recon && c < cone_end;
                if let Err(e) = dec.decode_sb(mi_row, c) {
                    return Err((row_key, e));
                }
                if dec.t.msac.overrun() {
                    return Err((row_key, Error::Truncated));
                }
                c += sb4;
            }
        }
        Ok(tile_s)
    }

    /// Called after superblock row `r` has been reconstructed.
    fn after_decode_row(&mut self, r: usize) -> Result<()> {
        let f = &self.f;
        // Save the unfiltered bottom row for intra prediction of row r + 1.
        for p in 0..f.num_planes {
            let sy = f.sub(p).1;
            let y = (((r + 1) * f.sb_px) >> sy) - 1;
            let y = y.min(self.s.planes[p].y0 + self.s.planes[p].rows - 1);
            self.s.above_line[p].copy_from_slice(self.s.planes[p].row(y));
            self.s.above_line_y[p] = y;
        }
        if self.deblock && r >= self.post_start {
            deblock::deblock_rows(
                f,
                &self.s.mi,
                &mut self.s.planes,
                r * f.sb4,
                (r + 1) * f.sb4,
                self.col_range,
                &self.dsp,
                &mut self.lf_grid,
            );
        }
        Ok(())
    }

    /// CDEF, superres and loop restoration after superblock row `r` has
    /// been deblocked, then emits all finished stripes. `last` is set for
    /// the final row.
    ///
    /// Like dav1d (`filter_sbrow_cdef`), the rows are post-filtered while
    /// they are still in the cache from deblocking: luma rows
    /// `post_row_start(r)..post_row_start(r + 1)`, i.e. all of row `r`
    /// except its last [`POST_LAG`] luma rows (which deblocking the top
    /// edge of row `r + 1` may change, or which CDEF reads as its bottom
    /// border), plus the last rows of row `r - 1`.
    fn post_row(&mut self, r: usize, last: bool, sink: &mut dyn RowSink<T>) -> Result<()> {
        let f = &self.f;
        let np = f.num_planes;
        let sb_px = f.sb_px;
        let lum0 = self.post_row_start(r);
        let lum1 = if last { (r + 1) * sb_px } else { self.post_row_start(r + 1) };
        // Plane row ranges of the step (clipped to the mi frame).
        let mut ranges = [(0usize, 0usize); 3];
        for p in 0..np {
            let sy = f.sub(p).1;
            let y0 = lum0 >> sy;
            let y1 = (lum1 >> sy).min(self.mi_h[p]);
            ranges[p] = (y0, y1);
        }
        // Save the deblocked rows around the LR stripe boundaries that CDEF
        // reaches in this step (all of them are final after deblocking row
        // `r`: boundary `k` covers luma rows `64k - 10..64k - 6`).
        if self.lr {
            let lim = (f.mi_rows * 4).min((r + 1) * sb_px);
            loop {
                let k = self.next_bound;
                if 64 * k - 8 >= lim || (!last && 64 * k - 10 >= lum1) {
                    break;
                }
                let mut bnd = self.lr_free.pop().unwrap_or_else(|| LrBoundary {
                    k: 0,
                    rows: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
                });
                bnd.k = k;
                for p in 0..np {
                    if f.fh.lr.frame_restoration_type[p] == RestorationType::None {
                        continue;
                    }
                    let sy = f.sub(p).1;
                    let b = (64 * k - 8) >> sy;
                    let w = self.out_w[p];
                    for (i, dst) in bnd.rows[p].iter_mut().enumerate() {
                        let y = (b + i - 2).min(self.mi_h[p] - 1);
                        let src = self.s.planes[p].row(y);
                        if self.superres {
                            dst.resize(self.up[p].stride, T::default());
                            self.upscalers[p].upscale_row(dst, src, f.bd);
                        } else {
                            dst.resize(w, T::default());
                            dst.copy_from_slice(&src[..w]);
                        }
                    }
                }
                self.lr_bounds.push_back(bnd);
                self.next_bound += 1;
            }
        }
        let row_end_luma = self.rows_end_mi_aligned();
        if self.cdef {
            let cur = self.cdef_cur;
            if r == self.post_start && r > 0 {
                // No CDEF ran for the rows above: use their current samples.
                for p in 0..np {
                    let y0 = ranges[p].0;
                    let w = self.mi_w[p];
                    for i in 0..2 {
                        self.cdef_line[cur][p][i]
                            .copy_from_slice(&self.s.planes[p].row(y0 - 2 + i)[..w]);
                    }
                }
            }
            let mut dims = [(0usize, 0usize); 3];
            for p in 0..np {
                let y1 = ranges[p].1;
                let w = self.mi_w[p];
                dims[p] = (
                    w,
                    self.mi_h[p].min((row_end_luma + f.sub(p).1) >> f.sub(p).1),
                );
                // Save the pre-CDEF bottom two rows for the next step.
                if !last && y1 >= 2 {
                    for i in 0..2 {
                        self.cdef_line[cur ^ 1][p][i]
                            .copy_from_slice(&self.s.planes[p].row(y1 - 2 + i)[..w]);
                    }
                }
            }
            let s = &mut self.s;
            let cdef_idx = &s.cdef_idx;
            let cs = s.cdef_stride;
            cdef_apply::cdef_sbrow(
                f,
                &s.mi,
                |r, c| cdef_idx[(r >> 4) * cs + (c >> 4)],
                &mut s.planes,
                &mut self.cdef_srcs,
                &self.cdef_line[cur],
                &ranges[..np],
                &dims[..np],
                lum0 / 4,
                lum1 / 4,
                self.col_range,
                &self.dsp,
            );
            self.cdef_cur ^= 1;
        }
        if self.superres {
            for p in 0..np {
                let (y0, y1) = ranges[p];
                for y in y0..y1 {
                    let (src, dst) = (&self.s.planes[p], &mut self.up[p]);
                    let i = dst.idx(0, y);
                    let stride = dst.stride;
                    self.upscalers[p].upscale_row(
                        &mut dst.data_mut()[i..i + stride],
                        src.row(y),
                        f.bd,
                    );
                }
            }
        }
        // Emit finished stripes.
        let done_luma = if last { usize::MAX } else { lum1 };
        let frame_h = f.fh.frame_height.min(self.rows_end);
        loop {
            let k = self.next_stripe;
            let start = (64 * k).saturating_sub(8);
            if start >= frame_h {
                break;
            }
            let end = (64 * k + 56).min(frame_h);
            if end > done_luma {
                break;
            }
            self.emit_stripe(k, sink)?;
            self.next_stripe += 1;
            while let Some(b) = self.lr_bounds.front() {
                if b.k < self.next_stripe {
                    self.lr_free.push(self.lr_bounds.pop_front().unwrap());
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    fn emit_stripe(&mut self, k: usize, sink: &mut dyn RowSink<T>) -> Result<()> {
        let f = &self.f;
        let np = f.num_planes;
        let mut geoms = [StripeGeom {
            start_y: 0,
            end_y: 0,
            plane_w: 0,
            plane_h: 0,
        }; 3];
        for p in 0..np {
            let sy = f.sub(p).1;
            let start_y = (64 * k as isize - 8) >> sy;
            let end_y = start_y + (64 >> sy) - 1;
            let plane_h = self.out_h[p].min((self.rows_end + sy) >> sy);
            geoms[p] = StripeGeom {
                start_y,
                end_y,
                plane_w: self.out_w[p],
                plane_h,
            };
        }
        let lr_on =
            |p: usize| self.lr && f.fh.lr.frame_restoration_type[p] != RestorationType::None;
        if self.direct {
            // Restore in place; the planes are the output.
            for p in (0..np).filter(|&p| lr_on(p)) {
                let above = self.lr_bounds.iter().find(|b| b.k == k).map(|b| &b.rows[p]);
                let below = self
                    .lr_bounds
                    .iter()
                    .find(|b| b.k == k + 1)
                    .map(|b| &b.rows[p]);
                lr_apply::lr_stripe_in_place(
                    &self.s.lr[p],
                    &geoms[p],
                    &mut self.s.planes[p],
                    above,
                    below,
                    &mut self.lr_scratch,
                    &mut self.lr_left,
                    f.sub(p).1,
                    f.bd,
                    &self.dsp,
                );
            }
            // Intra block copy reads unfiltered samples of the whole frame.
            if !self.intrabc {
                let y_end = (64 * k + 56).min(geoms[0].plane_h);
                let strides: Vec<usize> = self.s.planes[..np].iter().map(|w| w.stride).collect();
                let mut refs: Vec<&mut [T]> = self.s.planes[..np]
                    .iter_mut()
                    .map(|w| w.data_mut())
                    .collect();
                sink.rows_final(&mut refs, &strides, y_end);
            }
            return Ok(());
        }
        for p in 0..np {
            if !lr_on(p) {
                continue;
            }
            let g = &geoms[p];
            let src = if self.superres { &self.up[p] } else { &self.s.planes[p] };
            let ow = self.out_w[p];
            let out = &mut self.lr_out[p];
            let above = self.lr_bounds.iter().find(|b| b.k == k).map(|b| &b.rows[p]);
            let below = self
                .lr_bounds
                .iter()
                .find(|b| b.k == k + 1)
                .map(|b| &b.rows[p]);
            lr_apply::lr_stripe(
                &self.s.lr[p],
                g,
                src,
                above,
                below,
                out,
                ow,
                &mut self.lr_scratch,
                f.sub(p).1,
                f.bd,
                (
                    ((self.col_range.0 * 4) >> f.sub(p).0) * self.out_w[p] / self.mi_w[p].max(1),
                    ((self.col_range.1.saturating_mul(4)) >> f.sub(p).0)
                        .saturating_mul(self.out_w[p])
                        / self.mi_w[p].max(1),
                ),
                &self.dsp,
            );
        }
        let mut bands = Vec::with_capacity(np);
        for p in 0..np {
            let g = &geoms[p];
            let y_first = g.start_y.max(0) as usize;
            let y_last = (g.end_y as usize).min(g.plane_h - 1);
            // Planes without loop restoration are handed out directly
            // from the window.
            let (data, stride) = if lr_on(p) {
                (&self.lr_out[p][..], self.out_w[p])
            } else {
                let src = if self.superres { &self.up[p] } else { &self.s.planes[p] };
                (&src.data()[src.idx(0, y_first)..], src.stride)
            };
            bands.push(PlaneRows {
                y0: y_first,
                h: y_last + 1 - y_first,
                w: self.out_w[p],
                data,
                stride,
            });
        }
        sink.push(&bands)
    }
}

#[allow(unused)]
fn unsupported() -> Error {
    Error::Unsupported("")
}
