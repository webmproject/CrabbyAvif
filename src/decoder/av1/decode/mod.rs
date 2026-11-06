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
pub mod stage;
pub mod state;
pub mod superres;

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Mutex;

use crate::decoder::av1::error::{Error, Result};
use crate::decoder::av1::headers::RestorationType;
use crate::decoder::av1::obu::FrameData;
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::tables::cdf::CdfContext;
use crate::decoder::av1::types::MI_SIZE;

use block::{SbDecoder, Scratch};
use buffers::{MiWin, PlaneWin};
use cdef_apply::CdefSrc;
use lr_apply::{LrBoundary, StripeGeom};
use state::{AboveCtx, FrameCtx, FrameState, IntraSetup, LeftCtx, LrPlane, TileState};
use superres::Upscaler;

/// Process-wide pools that keep decoder state and output planes between
/// `av1::decode` calls, so that repeated decodes (on any thread) do not
/// reallocate and fault in frame sized buffers.
///
/// `DECODER_POOL` holds up to 16 `PooledState` per sample type and
/// `PLANE_POOL` up to 16 luma and 32 chroma planes per sample type (only
/// filled by `recycle_planes`). The limits count entries, not bytes: the
/// buffers are sized for the largest frame decoded so far and stay
/// allocated for the lifetime of the process.
static DECODER_POOL: Mutex<Vec<Box<dyn Any + Send>>> = Mutex::new(Vec::new());
static PLANE_POOL: Mutex<Vec<Box<dyn Any + Send>>> = Mutex::new(Vec::new());
/// Number of luma planes in `PLANE_POOL` (lets the common case of an empty
/// pool skip the lock).
static PLANE_POOL_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct BandWorkerScratch<T: Pixel> {
    cdef_srcs: [CdefSrc; 3],
    scratch: Vec<T>,
    lr_band_out: [Vec<T>; 3],
    /// Deblocking edge parameter scratch of the vertical edge pass.
    lf_grid: Vec<u8>,
}

impl<T: Pixel> Default for BandWorkerScratch<T> {
    fn default() -> Self {
        Self {
            cdef_srcs: Default::default(),
            scratch: Vec::new(),
            lr_band_out: Default::default(),
            lf_grid: Vec::new(),
        }
    }
}

struct PooledState<T: Pixel> {
    s: FrameState<T>,
    sc: Scratch,
    non_direct_planes: [PlaneWin<T>; 3],
    cdfs: Vec<Box<CdfContext>>,
    stub_cdf: Option<Box<CdfContext>>,
    cdef_line: [[[Vec<T>; 2]; 3]; 2],
    cdef_srcs: [CdefSrc; 3],
    lr_bounds: VecDeque<LrBoundary<T>>,
    lr_free: Vec<LrBoundary<T>>,
    lr_out: [Vec<T>; 3],
    lr_scratch: Vec<T>,
    lf_grid: Vec<u8>,
    band_free: Vec<Box<BandBufs<T>>>,
    band_carry: [PlaneWin<T>; 3],
    band_scratches: Vec<BandWorkerScratch<T>>,
    order_carry: MiWin,
    tile_workers: Vec<TileWorker<T>>,
    recon_sc: Scratch,
    pipe_chunks_free: Vec<stage::LogChunk>,
}

impl<T: Pixel> Default for PooledState<T> {
    fn default() -> Self {
        Self {
            s: FrameState::default(),
            sc: Scratch::new(),
            non_direct_planes: Default::default(),
            cdfs: Vec::new(),
            stub_cdf: None,
            cdef_line: Default::default(),
            cdef_srcs: Default::default(),
            lr_bounds: VecDeque::new(),
            lr_free: Vec::new(),
            lr_out: Default::default(),
            lr_scratch: Vec::new(),
            lf_grid: Vec::new(),
            band_free: Vec::new(),
            band_carry: Default::default(),
            band_scratches: Vec::new(),
            order_carry: MiWin::default(),
            tile_workers: Vec::new(),
            recon_sc: Scratch::new(),
            pipe_chunks_free: Vec::new(),
        }
    }
}

struct TypedPlanePool<T> {
    luma: Vec<Vec<T>>,
    chroma: Vec<Vec<T>>,
}

fn pop_direct_planes<T: Pixel>(np: usize) -> [Vec<T>; 3] {
    if PLANE_POOL_LEN.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return Default::default();
    }
    let mut bufs: [Vec<T>; 3] = Default::default();
    if let Ok(mut guard) = PLANE_POOL.lock() {
        if let Some(pool) = guard
            .iter_mut()
            .find_map(|b| b.downcast_mut::<TypedPlanePool<T>>())
        {
            if let Some(y) = pool.luma.pop() {
                bufs[0] = y;
                PLANE_POOL_LEN.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
            for dst in &mut bufs[1..np.min(3)] {
                if let Some(uv) = pool.chroma.pop() {
                    *dst = uv;
                }
            }
        }
    }
    bufs
}

pub(super) fn recycle_planes<T: Pixel>(planes: Vec<crate::decoder::av1::sink::Plane<T>>) {
    if planes.is_empty() {
        return;
    }
    let mut bufs: [Vec<T>; 3] = Default::default();
    for (i, p) in planes.into_iter().take(3).enumerate() {
        bufs[i] = p.data;
    }
    recycle_raw_planes(bufs);
}

pub(super) fn recycle_raw_planes<T: Pixel>(bufs: [Vec<T>; 3]) {
    if bufs.iter().all(|b| b.is_empty()) {
        return;
    }
    if let Ok(mut guard) = PLANE_POOL.lock() {
        let pos = guard
            .iter()
            .position(|b| b.is::<TypedPlanePool<T>>())
            .unwrap_or_else(|| {
                guard.push(Box::new(TypedPlanePool::<T> {
                    luma: Vec::with_capacity(16),
                    chroma: Vec::with_capacity(32),
                }));
                guard.len() - 1
            });
        let Some(pool) = guard[pos].downcast_mut::<TypedPlanePool<T>>() else {
            return;
        };
        for (i, data) in bufs.into_iter().enumerate() {
            if data.is_empty() {
                continue;
            }
            if i == 0 {
                if pool.luma.len() < 16 {
                    pool.luma.push(data);
                    PLANE_POOL_LEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            } else if pool.chroma.len() < 32 {
                pool.chroma.push(data);
            }
        }
    }
}

fn pop_pooled_state<T: Pixel>() -> Box<PooledState<T>> {
    if let Ok(mut pool) = DECODER_POOL.lock() {
        if let Some(pos) = pool.iter().rposition(|b| b.is::<PooledState<T>>()) {
            if let Ok(state) = pool.swap_remove(pos).downcast::<PooledState<T>>() {
                return state;
            }
        }
    }
    Box::default()
}

fn push_pooled_state<T: Pixel>(state: Box<PooledState<T>>) {
    if let Ok(mut pool) = DECODER_POOL.lock() {
        if pool.len() < 16 {
            pool.push(state);
        }
    }
}

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

    /// Whether [`RowSink::rows_final`] should be called after each stripe.
    fn wants_rows_final(&self) -> bool {
        false
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
    s: &'a mut FrameState<T>,
    sc: &'a mut Scratch,
    cdfs: &'a mut Vec<Box<CdfContext>>,
    stub_cdf: &'a mut Option<Box<CdfContext>>,
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
    cdef_line: &'a mut [[[Vec<T>; 2]; 3]; 2],
    cdef_cur: usize,
    cdef_srcs: &'a mut [CdefSrc; 3],
    up: Vec<PlaneWin<T>>,
    upscalers: Vec<Upscaler>,
    lr_bounds: &'a mut VecDeque<LrBoundary<T>>,
    lr_free: &'a mut Vec<LrBoundary<T>>,
    next_stripe: usize,
    /// Next LR stripe boundary whose deblocked rows are to be saved.
    next_bound: usize,
    lr_out: &'a mut [Vec<T>; 3],
    lr_out_planes: [PlaneWin<T>; 3],
    lr_scratch: &'a mut Vec<T>,
    /// Deblocking edge parameter scratch.
    lf_grid: &'a mut Vec<u8>,
    band_free: &'a mut Vec<Box<BandBufs<T>>>,
    band_carry: &'a mut [PlaneWin<T>; 3],
    band_scratches: &'a mut Vec<BandWorkerScratch<T>>,
    order_carry: &'a mut MiWin,
    tile_workers: &'a mut Vec<TileWorker<T>>,
    mi_arrays: (bool, bool, bool),
    intrabc: bool,
    /// Decoding into full frame planes that are handed to the sink at
    /// the end (see `RowSink::wants_frame`).
    direct: bool,
    /// Post-filtering on worker threads (see `Pipeline::run_bands`).
    bands: bool,
    /// Pre loop restoration samples of the last columns of the previous
    /// restoration unit (in-place loop restoration).
    /// DSP kernel table.
    dsp: crate::decoder::av1::dsp::Dsp<T>,
    /// First superblock row that is post-filtered and output.
    post_start: usize,
    /// Superblock row end of the decoded region.
    row_end: usize,
    /// Mode-info column range that needs post-filtering.
    col_range: (usize, usize),
    /// Luma rows `0..rows_end` are decoded (the whole frame unless a region
    /// is decoded).
    rows_end: usize,
    recon_sc: &'a mut Scratch,
    pipe_chunks_free: &'a mut Vec<stage::LogChunk>,
    direct_bufs: [Vec<T>; 3],
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
    let mut pooled = pop_pooled_state::<T>();
    let mut mi_w = [0; 3];
    let mut mi_h = [0; 3];
    let mut out_w = [0; 3];
    let mut out_h = [0; 3];
    // Without intra block copy only rows from the superblock row above
    // (minus 16) on are needed (2 * sb_px + 16 rows); the spare rows
    // make the window move (copy) only every third superblock row.
    let deblock =
        params.filters.deblock && !intrabc && (fh.lf.level[0] != 0 || fh.lf.level[1] != 0);
    let deblock_uv = deblock && np > 1 && (fh.lf.level[2] != 0 || fh.lf.level[3] != 0);
    let cdef = params.filters.cdef
        && !intrabc
        && f.seq.enable_cdef
        && !fh.coded_lossless
        && cdef_active(fh);
    let need_cdef_idx = cdef;
    let lr = params.filters.restoration && fh.lr.uses_lr;
    let full_out = params.region.is_none() && !fh.use_superres && sink.wants_frame();
    // Frames with post filters: the filters run on other threads (single
    // tile frames with CDEF or loop restoration, see `Pipeline::run_bands`;
    // multi-tile frames with any post filter, see
    // `Pipeline::run_tiles_bands`).
    let single_tile = fh.tile_info.rows == 1 && fh.tile_info.cols == 1;
    let bands =
        full_out && params.threads > 1 && !intrabc && f.sb_rows > 1 && (cdef || lr || deblock);
    let tiles_bands = bands && !single_tile;
    let direct = full_out && !bands;
    let has_post = deblock || cdef || fh.lr.uses_lr || fh.use_superres || sink.wants_rows_final();
    let tiles_direct = direct && !single_tile && !has_post;
    let tiles_mt = tiles_bands || tiles_direct;
    // In `run_bands` the rows above a superblock row are filled into its
    // row buffer only after the row has been decoded.
    let need_above_line = f.sb_rows > 1 && (deblock || cdef || bands);
    let win_rows_luma = if intrabc || direct {
        mi_rows_al * 4
    } else if bands {
        sb_px + BAND_TOP
    } else {
        4 * sb_px + 16
    };
    let mut direct_bufs = if direct { pop_direct_planes::<T>(np) } else { Default::default() };
    let mut lr_out_planes: [PlaneWin<T>; 3] = Default::default();
    let direct_pipe_split = direct
        && single_tile
        && params.threads > 1
        && !intrabc
        && !fh.use_superres
        && f.sb_rows > 1
        && !has_post;
    for p in 0..np {
        let (sx, sy) = f.sub(p);
        mi_w[p] = (f.mi_cols * 4) >> sx;
        mi_h[p] = (f.mi_rows * 4) >> sy;
        out_w[p] = (fh.upscaled_width + sx) >> sx;
        out_h[p] = (fh.frame_height + sy) >> sy;
        let stride = ((mi_cols_al * 4) >> sx).next_multiple_of(64);
        let lr_p = lr && !intrabc && fh.lr.frame_restoration_type[p] != RestorationType::None;
        if direct && !lr_p {
            if !direct_pipe_split {
                let buf = std::mem::take(&mut direct_bufs[p]);
                pooled.s.planes[p] = PlaneWin::from_vec(buf, stride, win_rows_luma >> sy);
            }
        } else if tiles_bands {
            // Only the stride is used (by `band_setup`); the tile columns
            // decode into their own windows. Keep the window consistent
            // (no rows) while the pooled allocation is kept.
            std::mem::swap(&mut pooled.s.planes[p], &mut pooled.non_direct_planes[p]);
            pooled.s.planes[p].reset(stride, 0, 0, 0);
        } else {
            std::mem::swap(&mut pooled.s.planes[p], &mut pooled.non_direct_planes[p]);
            let rows = if intrabc {
                (mi_rows_al * 4) >> sy
            } else if bands {
                (sb_px + BAND_TOP) >> sy
            } else {
                (4 * sb_px + 16) >> sy
            };
            pooled.s.planes[p].reset(stride, rows, 0, 0);
            if direct && lr_p {
                let buf = std::mem::take(&mut direct_bufs[p]);
                lr_out_planes[p] = PlaneWin::from_vec(buf, stride, (mi_rows_al * 4) >> sy);
            }
        }
        if !tiles_mt {
            if need_above_line {
                pooled.s.above_line[p].resize(stride + 64, T::default());
            } else {
                pooled.s.above_line[p].clear();
            }
        }
    }
    let mi_arrays = (deblock, deblock_uv, deblock && fh.delta_lf_present);
    if !tiles_mt {
        let above_w = mi_cols_al + 32;
        pooled.s.mi.reset(
            mi_cols_al + 32,
            2 * f.sb4,
            0,
            0,
            f.ss_x,
            f.ss_y,
            deblock,
            deblock_uv,
            deblock && fh.delta_lf_present,
            intrabc,
        );
        pooled
            .s
            .above
            .reset(0, above_w, np, fh.allow_screen_content_tools);
    }
    if need_cdef_idx {
        let cdef_len = (mi_rows_al / 16 + 2) * (mi_cols_al / 16 + 2);
        if params.region.is_some() {
            let pl = pooled.s.cdef_idx.len().min(cdef_len);
            pooled.s.cdef_idx[..pl].fill(-1);
        }
        pooled.s.cdef_idx.resize(cdef_len, -1);
    } else {
        pooled.s.cdef_idx.clear();
    }
    pooled.s.cdef_stride = mi_cols_al / 16 + 2;
    pooled.s.cdef_row0 = 0;
    pooled.s.cdef_col0 = 0;
    if fh.lr.uses_lr {
        for (p, lp) in pooled.s.lr.iter_mut().enumerate() {
            if p < np {
                lp.reset(fh, p, f.ss_x, f.ss_y);
            } else {
                lp.units.clear();
            }
        }
    }
    if !tiles_mt {
        pooled.s.ipred = [IntraSetup::new(&f, 0), IntraSetup::new(&f, 1)];
    }
    let superres = fh.use_superres;
    if cdef && !bands {
        for bufs in &mut pooled.cdef_line {
            for (pl, pair) in bufs[..np].iter_mut().enumerate() {
                pair[0].resize(mi_w[pl], T::default());
                pair[1].resize(mi_w[pl], T::default());
            }
        }
    }
    if lr && !direct && !bands {
        for (p, out) in pooled.lr_out[..np].iter_mut().enumerate() {
            if fh.lr.frame_restoration_type[p] != RestorationType::None {
                out.resize(out_w[p] * (64 >> f.sub(p).1), T::default());
            } else {
                out.clear();
            }
        }
    }
    let res = {
        let PooledState {
            s,
            sc,
            cdfs,
            stub_cdf,
            cdef_line,
            cdef_srcs,
            lr_bounds,
            lr_free,
            lr_out,
            lr_scratch,
            lf_grid,
            band_free,
            band_carry,
            band_scratches,
            order_carry,
            tile_workers,
            recon_sc,
            pipe_chunks_free,
            ..
        } = &mut *pooled;
        let mut p = Pipeline {
            deblock,
            cdef,
            lr,
            superres,
            out_w,
            out_h,
            mi_w,
            mi_h,
            cdef_line,
            cdef_cur: 0,
            cdef_srcs,
            up: Vec::new(),
            upscalers: Vec::new(),
            lr_bounds,
            lr_free,
            next_stripe: 0,
            next_bound: 1,
            lr_out,
            lr_out_planes,
            lr_scratch,
            lf_grid,
            band_free,
            band_carry,
            band_scratches,
            order_carry,
            tile_workers,
            mi_arrays,
            intrabc,
            direct,
            bands,
            dsp: crate::decoder::av1::dsp::Dsp::new(f.bd),
            post_start: 0,
            row_end: 0,
            col_range: (0, usize::MAX),
            rows_end: usize::MAX,
            s,
            sc,
            cdfs,
            stub_cdf,
            f,
            recon_sc,
            pipe_chunks_free,
            direct_bufs,
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
    };
    if res.is_ok() {
        for p in 0..np {
            let lr_p = lr && !intrabc && fh.lr.frame_restoration_type[p] != RestorationType::None;
            if !direct || lr_p {
                std::mem::swap(&mut pooled.s.planes[p], &mut pooled.non_direct_planes[p]);
            }
        }
        if !pooled.lr_bounds.is_empty() {
            pooled.lr_free.extend(pooled.lr_bounds.drain(..));
        }
        push_pooled_state(pooled);
    }
    res
}

/// Byte threshold at which Stage P flushes an in-progress `LogChunk` to
/// Stage R mid-row. Keeping chunks small (~8 KiB) lets Stage R overlap
/// with Stage P within the first and last superblock rows (and on small
/// direct-pipe frames) while keeping recycled chunk buffers in L2 cache.
const LOG_CHUNK_FLUSH_BYTES: usize = 8 * 1024;

/// Process-wide pool of persistent background worker threads (up to 16),
/// avoiding OS thread creation/teardown, `mmap_lock` contention, and 0-PELT
/// Little-core placement across repeated single-tile `run_direct_pipe` and
/// `run_bands_pipe` decodes.
struct PersistentWorker {
    tx: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
}

static WORKER_POOL: Mutex<Vec<PersistentWorker>> = Mutex::new(Vec::new());

fn pop_persistent_worker() -> PersistentWorker {
    if let Ok(mut pool) = WORKER_POOL.lock() {
        if let Some(w) = pool.pop() {
            return w;
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
    std::thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
        }
    });
    PersistentWorker { tx }
}

fn push_persistent_worker(w: PersistentWorker) {
    if let Ok(mut pool) = WORKER_POOL.lock() {
        if pool.len() < 16 {
            pool.push(w);
        }
    }
}

struct WorkerDoneGuard(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for WorkerDoneGuard {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

struct ActiveWorkerGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl Drop for ActiveWorkerGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Ensures `handoff_state` reaches a terminal state (`2` handed off or `3`
/// finished/aborted without handoff) even if the worker exits early or panics.
struct HandoffGuard(std::sync::Arc<std::sync::atomic::AtomicU8>);

impl Drop for HandoffGuard {
    fn drop(&mut self) {
        let mut cur = self.0.load(std::sync::atomic::Ordering::Relaxed);
        while cur < 2 {
            match self.0.compare_exchange_weak(
                cur,
                3,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
}

/// Stage R reconstruction state transferred from a worker thread back to the
/// main (Prime-core) thread once Stage P finishes entropy parsing.
struct ReconHandoff<T: Pixel> {
    first: stage::PipeMsg<T>,
    rx: std::sync::mpsc::Receiver<stage::PipeMsg<T>>,
    cursor: stage::LogCursor,
    stub_ts: TileState<'static>,
    r_state: FrameState<T>,
    r_sc: Scratch,
}

#[allow(clippy::too_many_arguments)]
fn step_bands_recon_msg<T: Pixel>(
    msg: stage::PipeMsg<T>,
    cursor: &mut stage::LogCursor,
    f: &FrameCtx,
    stub_ts: &mut TileState,
    r_state: &mut FrameState<T>,
    r_sc: &mut Scratch,
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
    sb4: usize,
    sb_px: usize,
    sb_rows: usize,
    bp: &BandParams,
    pipe: &BandPipeline<T>,
    r_w: &mut BandWorker<'_, T>,
    backlog_limit: usize,
    recycle_tx: &std::sync::mpsc::Sender<stage::LogChunk>,
) -> Result<()> {
    match msg {
        stage::PipeMsg::Chunk(mut chunk) => {
            decode_log_chunk(&chunk, cursor, f, stub_ts, r_state, r_sc, dsp, sb4)?;
            chunk.clear();
            let _ = recycle_tx.send(chunk);
            if pipe.queued_tasks() > backlog_limit {
                pipe.work(r_w, bp, true, None);
            }
        }
        stage::PipeMsg::RowEnd {
            r,
            last,
            bufs,
            chunk,
        } => {
            if let Some(mut chunk) = chunk {
                decode_log_chunk(&chunk, cursor, f, stub_ts, r_state, r_sc, dsp, sb4)?;
                chunk.clear();
                let _ = recycle_tx.send(chunk);
                if pipe.queued_tasks() > backlog_limit {
                    pipe.work(r_w, bp, true, None);
                }
            }
            if r + 1 < sb_rows && !r_state.above_line[0].is_empty() {
                for p in 0..f.num_planes {
                    let sy = f.sub(p).1;
                    let y = (((r + 1) * f.sb_px) >> sy) - 1;
                    let y = y.min(r_state.planes[p].y0 + r_state.planes[p].rows - 1);
                    let row = &r_state.planes[p].row(y)[..bp.mi_w[p]];
                    r_state.above_line[p][..row.len()].copy_from_slice(row);
                }
            }
            let Some(mut bb) = bufs else {
                return Err(Error::Invalid("missing bufs in RowEnd"));
            };
            for p in 0..f.num_planes {
                let sy = f.sub(p).1;
                if !last {
                    let stride = r_state.planes[p].stride;
                    bb.planes[p].reset(
                        stride,
                        (sb_px + BAND_TOP) >> sy,
                        0,
                        ((r + 1) * sb_px - BAND_TOP) >> sy,
                    );
                }
                std::mem::swap(&mut r_state.planes[p], &mut bb.planes[p]);
            }
            let job = RowJob { r, last, bufs: bb };
            if pipe.push_row(job) > backlog_limit && !last {
                pipe.work(r_w, bp, true, None);
            }
        }
    }
    Ok(())
}

fn decode_log_chunk<T: Pixel>(
    chunk: &stage::LogChunk,
    cursor: &mut stage::LogCursor,
    f: &FrameCtx,
    stub_ts: &mut TileState,
    r_state: &mut FrameState<T>,
    r_sc: &mut Scratch,
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
    sb4: usize,
) -> Result<()> {
    cursor.reset();
    let sb_mask = !(sb4 - 1);
    let mut prev_sb_y = usize::MAX;
    let mut b = block::Block::default();
    for &(mi_row, mi_col, num_blocks) in &chunk.sbs {
        let sb_y = (mi_row & sb_mask) << 2;
        if sb_y != prev_sb_y {
            prev_sb_y = sb_y;
            r_state.set_sb_y(f, sb_y);
        }
        let mut dec = SbDecoder {
            f,
            t: stub_ts,
            s: r_state,
            sc: r_sc,
            dsp,
            b,
            recon: true,
            sb_y,
            read_deltas: false,
            plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
            stage: stage::DecoderStage::Recon(chunk, cursor),
            filter_type: [false; 2],
        };
        dec.clear_block_decoded_flags(mi_row, mi_col);
        for _ in 0..num_blocks {
            dec.decode_recon_block(chunk)?;
        }
        b = dec.b;
    }
    if cursor.block_idx != chunk.blocks.len()
        || cursor.palette_idx != chunk.palettes.len()
        || cursor.tx_header_idx != chunk.tx_headers.len()
        || cursor.tx_entry_idx != chunk.tx_entries.len()
    {
        return Err(Error::Invalid("pipeline chunk length mismatch"));
    }
    Ok(())
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
        if ti.rows == 1 && ti.cols == 1 && params.region.is_none() {
            if self.bands {
                return self.run_bands(fd, params.threads, sink);
            }
            let use_split = params.threads >= 2
                && !self.intrabc
                && !fh.use_superres
                && sb_rows >= 2
                && self.direct
                && !self.has_post_filters()
                && !sink.wants_rows_final();
            if use_split {
                return self.run_direct_pipe(fd, sink);
            }
            if self.s.planes[0].is_empty() {
                let mi_cols_al = f.sb_cols * f.sb4;
                let mi_rows_al = f.sb_rows * f.sb4;
                let win_rows_luma = mi_rows_al * 4;
                for p in 0..f.num_planes {
                    let (sx, sy) = f.sub(p);
                    let stride = ((mi_cols_al * 4) >> sx).next_multiple_of(64);
                    let buf = std::mem::take(&mut self.direct_bufs[p]);
                    self.s.planes[p] = PlaneWin::from_vec(buf, stride, win_rows_luma >> sy);
                }
            }
            self.post_start = 0;
            self.row_end = sb_rows;
            self.col_range = (0, f.sb_cols * sb4);
            self.next_stripe = 0;
            self.next_bound = 1;
            self.rows_end = fh.frame_height;
            let c0 = ti.mi_col_starts[0];
            let c1 = ti.mi_col_starts[1];
            let mut ts = TileState::with_cdf(
                fd.single_tile,
                fh,
                f.seq,
                (ti.mi_row_starts[0], ti.mi_row_starts[1]),
                (c0, c1),
                self.cdfs.pop(),
            );
            let mut b = block::Block::default();
            for r in 0..sb_rows {
                let mi_row = r * sb4;
                self.slide_windows(r);
                self.s.left.clear();
                let sb_y = r * self.f.sb_px;
                self.s.set_sb_y(&self.f, sb_y);
                {
                    let mut dec = SbDecoder {
                        f: &self.f,
                        t: &mut ts,
                        s: self.s,
                        sc: self.sc,
                        dsp: &self.dsp,
                        b,
                        recon: true,
                        sb_y,
                        read_deltas: false,
                        plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
                        stage: stage::DecoderStage::Fused,
                        filter_type: [false; 2],
                    };
                    let mut c = c0;
                    while c < c1 {
                        dec.decode_sb(mi_row, c)?;
                        if dec.t.msac.overrun() {
                            return Err(Error::Truncated);
                        }
                        c += sb4;
                    }
                    b = dec.b;
                }
                self.after_decode_row(r)?;
                self.post_row(r, r + 1 == sb_rows, sink)?;
            }
            self.cdfs.push(ts.cdf);
            if self.direct {
                self.finish_direct(sink)?;
            }
            return Ok(());
        }
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
        self.row_end = row_end;
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
        if tile_jobs.len() > 1
            && params.region.is_none()
            && self.direct
            && !self.has_post_filters()
            && !sink.wants_rows_final()
        {
            if params.threads > 1 {
                return self.run_tiles_direct(fd, params.threads, sink);
            }
            return self.run_tiles_direct_st(fd, sink);
        }
        if params.threads > 1 && tile_jobs.len() > 1 {
            if self.bands {
                return self.run_tiles_bands(fd, params.threads, sink);
            }
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
                let first_tr = cur_tile_row == usize::MAX;
                cur_tile_row = tr;
                for ts in tiles.drain(..).flatten() {
                    self.cdfs.push(ts.cdf);
                }
                for tc in 0..ti.cols {
                    if !needed[tr][tc] {
                        tiles.push(None);
                        continue;
                    }
                    let data = fd.tiles()[tr * ti.cols + tc];
                    let ts = TileState::with_cdf(
                        data,
                        fh,
                        self.f.seq,
                        (ti.mi_row_starts[tr], ti.mi_row_starts[tr + 1]),
                        (ti.mi_col_starts[tc], ti.mi_col_starts[tc + 1]),
                        self.cdfs.pop(),
                    );
                    tiles.push(Some(ts));
                }
                if !first_tr {
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
            }
            self.slide_windows(r);
            // Decode the superblock row of every needed tile column.
            let sb_y = r * self.f.sb_px;
            self.s.set_sb_y(&self.f, sb_y);
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
                    stage: stage::DecoderStage::Fused,
                    filter_type: [false; 2],
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
        for ts in tiles.drain(..).flatten() {
            self.cdfs.push(ts.cdf);
        }
        if self.direct {
            // Hand the planes over as the output frame.
            self.finish_direct(sink)?;
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
        if self.intrabc || (self.direct && !self.lr) {
            return;
        }
        // Rows still read once row `r` is decoded start 16 luma rows above
        // it: intra prediction reads row `r * sb_px - 1`, deblocking its
        // top edge reads 8 rows above, post filtering `post_row(r)` starts
        // at `r * sb_px - POST_LAG` and CDEF / loop restoration read up to
        // 3 rows above that. Keeping only those makes the window move
        // (copy) rarely and copy few rows.
        let y0 = (r * f.sb_px).saturating_sub(POST_LAG + 8);
        let need_end = (r + 1) * f.sb_px;
        let keep_end = r * f.sb_px;
        for (p, pl) in self.s.planes[..f.num_planes].iter_mut().enumerate() {
            if self.direct && f.fh.lr.frame_restoration_type[p] == RestorationType::None {
                continue;
            }
            let sy = f.sub(p).1;
            pl.slide_to(y0 >> sy, need_end >> sy, keep_end >> sy);
        }
        for (p, pl) in self.up.iter_mut().enumerate() {
            let sy = f.sub(p).1;
            pl.slide_to(y0 >> sy, need_end >> sy, keep_end >> sy);
        }
    }

    /// Whether any in-loop filter after reconstruction changes samples
    /// (CDEF with all strengths zero does not).
    fn has_post_filters(&self) -> bool {
        let fh = self.f.fh;
        self.deblock || (self.cdef && cdef_active(fh)) || fh.lr.uses_lr || self.superres
    }

    /// Single tile frames with CDEF and / or loop restoration (whole frame,
    /// output planes handed to the sink): this thread only decodes. Every
    /// superblock row is decoded into its own row buffer (luma rows
    /// `r * sb_px - BAND_TOP .. (r + 1) * sb_px`, intra prediction takes
    /// the row above from `above_line`), which is then queued as a
    /// [`RowJob`] with copies of the mode info, `cdef_idx` and restoration
    /// units it needs. A worker runs the sequential finishing stage on the
    /// queued rows in order ([`BandWorker::finish_row`]: fill the rows
    /// above from the previous row, deblock, save the stripe boundaries),
    /// which turns the row into a [`BandJob`] for the rows that are final
    /// after deblocking (`band_rows`). The bands are independent: the
    /// workers run CDEF on them and loop restoration (or a copy) into their
    /// own rows of the output planes, which are split into the bands with
    /// `split_at_mut`. When too much work is queued this thread helps.
    fn run_bands(
        &mut self,
        fd: &FrameData,
        threads: usize,
        sink: &mut dyn RowSink<T>,
    ) -> Result<()> {
        let sb_rows = self.f.sb_rows;
        let sb4 = self.f.sb4;
        let sb_px = self.f.sb_px;
        let fh = self.f.fh;
        let ti = &fh.tile_info;
        let use_split = threads >= 2 && !self.intrabc && !fh.use_superres && sb_rows >= 2;
        if use_split {
            return self.run_bands_pipe(fd, threads, sink);
        }
        let (bp, out_planes) = self.band_setup();
        // At most one worker per band after the first.
        let max_bands = if !bp.cdef && sb_px > 64 { sb_rows * 2 } else { sb_rows };
        let workers = (threads - 1).min(max_bands.saturating_sub(1));
        let (res, out_planes) = {
            let pipe = BandPipeline::with_pooled(
                out_planes,
                std::mem::take(self.band_free),
                std::mem::take(self.band_carry),
                std::mem::take(self.lf_grid),
                std::mem::take(self.lr_free),
                workers,
            );
            pipe.set_parallel_bands(bp.cdef || bp.lr);
            let mut worker_scratches: Vec<BandWorkerScratch<T>> = (0..workers)
                .map(|_| self.band_scratches.pop().unwrap_or_default())
                .collect();
            let main_sc = self.band_scratches.pop().unwrap_or_default();
            let res = std::thread::scope(|scope| -> Result<()> {
                // If this thread panics (e.g. while decoding), the workers
                // must not keep waiting for rows that never come: the scope
                // joins them before the panic propagates.
                let _abort = AbortOnPanic(&pipe);
                let mut handles = Vec::with_capacity(workers);
                for sc in worker_scratches.drain(..) {
                    let (pipe, bp) = (&pipe, &bp);
                    handles.push(scope.spawn(move || {
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                            let _abort = AbortOnPanic(pipe);
                            let mut w = BandWorker::<T>::with_scratch(&fd.seq, &fd.frame, sc);
                            pipe.work(&mut w, bp, false, None);
                            pipe.return_scratch(w.into_scratch());
                        }));
                    }));
                }
                let mut w = BandWorker::<T>::with_scratch(&fd.seq, &fd.frame, main_sc);
                let (c0, c1) = (ti.mi_col_starts[0], ti.mi_col_starts[1]);
                let mut ts = TileState::with_cdf(
                    fd.single_tile,
                    fh,
                    self.f.seq,
                    (ti.mi_row_starts[0], ti.mi_row_starts[1]),
                    (c0, c1),
                    self.cdfs.pop(),
                );
                let mut b = block::Block::default();
                let res = 'dec: {
                    for r in 0..sb_rows {
                        let mi_row = r * sb4;
                        self.s.mi.slide_to(mi_row.saturating_sub(sb4));
                        self.s.left.clear();
                        let sb_y = r * sb_px;
                        self.s.set_sb_y(&self.f, sb_y);
                        let mut dec = SbDecoder {
                            f: &self.f,
                            t: &mut ts,
                            s: self.s,
                            sc: self.sc,
                            dsp: &self.dsp,
                            b,
                            recon: true,
                            sb_y,
                            read_deltas: false,
                            plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
                            stage: stage::DecoderStage::Fused,
                            filter_type: [false; 2],
                        };
                        let mut c = c0;
                        while c < c1 {
                            if let Err(e) = dec.decode_sb(mi_row, c) {
                                break 'dec Err(e);
                            }
                            if dec.t.msac.overrun() {
                                break 'dec Err(Error::Truncated);
                            }
                            c += sb4;
                        }
                        b = dec.b;
                        self.save_above_line(r);
                        let job = self.capture_row(r, r + 1 == sb_rows, &bp, pipe.take_bufs());
                        if pipe.push_row(job) > workers {
                            // Too much queued: help instead of decoding ahead.
                            pipe.work(&mut w, &bp, true, None);
                        }
                    }
                    Ok(())
                };
                self.cdfs.push(ts.cdf);
                pipe.close();
                if res.is_err() {
                    pipe.abort();
                    while handles.iter().any(|h| !h.is_finished()) {
                        for _ in 0..64 {
                            std::hint::spin_loop();
                        }
                    }
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                } else {
                    // Help with what is still queued without sleeping.
                    pipe.work_spin(&mut w, &bp, None, || {
                        handles.iter().all(|h| h.is_finished())
                    });
                }
                pipe.return_scratch(w.into_scratch());
                drop(handles);
                res?;
                if pipe.is_aborted() {
                    return Err(Error::Invalid("decoding thread panicked"));
                }
                Ok(())
            });
            let out_planes = pipe.reclaim_into(
                self.band_free,
                self.band_carry,
                self.lf_grid,
                self.lr_free,
                self.band_scratches,
            );
            (res, out_planes)
        };
        res?;
        self.finish_bands(out_planes, sink)
    }

    fn run_bands_pipe(
        &mut self,
        fd: &FrameData,
        threads: usize,
        sink: &mut dyn RowSink<T>,
    ) -> Result<()> {
        let sb_rows = self.f.sb_rows;
        let sb4 = self.f.sb4;
        let sb_px = self.f.sb_px;
        let fh = self.f.fh;
        let ti = &fh.tile_info;
        let (bp, out_planes) = self.band_setup();
        let max_bands = if !bp.cdef && sb_px > 64 { sb_rows * 2 } else { sb_rows };
        let workers = (threads.saturating_sub(2)).min(max_bands.saturating_sub(1));

        // Unbounded channel so Stage P on the main thread never sleeps on
        // backpressure (sleeping decays PELT utilization on heterogeneous
        // ARM big.LITTLE cores and causes wakeup migration off the Prime
        // core while mid cores are frequency-capped below misfit).
        let (tx, rx) = std::sync::mpsc::channel::<stage::PipeMsg<T>>();
        let (recycle_tx, recycle_rx) = std::sync::mpsc::channel::<stage::LogChunk>();

        let r_planes = std::mem::take(&mut self.s.planes);
        let r_above_line = std::mem::take(&mut self.s.above_line);
        let mut r_sc = std::mem::take(self.recon_sc);
        let (c0, c1) = (ti.mi_col_starts[0], ti.mi_col_starts[1]);

        let pipe = std::sync::Arc::new(BandPipeline::with_pooled(
            out_planes,
            std::mem::take(self.band_free),
            std::mem::take(self.band_carry),
            std::mem::take(self.lf_grid),
            std::mem::take(self.lr_free),
            workers,
        ));
        pipe.set_parallel_bands(bp.cdef || bp.lr);
        let shared = std::sync::Arc::new((self.f.seq.clone(), self.f.fh.clone(), bp));
        type ReconOut<T> = Result<([PlaneWin<T>; 3], [Vec<T>; 3], Scratch, Box<CdfContext>)>;
        let r_slot: std::sync::Arc<Mutex<Option<ReconOut<T>>>> =
            std::sync::Arc::new(Mutex::new(None));
        let handoff_state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
        let handoff_slot: std::sync::Arc<Mutex<Option<ReconHandoff<T>>>> =
            std::sync::Arc::new(Mutex::new(None));
        let active_workers = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(workers + 1));

        let mut pws = Vec::with_capacity(workers + 1);
        for _ in 0..workers {
            let pw = pop_persistent_worker();
            let pipe_w = pipe.clone();
            let shared_w = shared.clone();
            let sc_w = self.band_scratches.pop().unwrap_or_default();
            let active_guard = ActiveWorkerGuard(active_workers.clone());
            let _ = pw.tx.send(Box::new(move || {
                let _active = active_guard;
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    let _abort = AbortOnPanic(&pipe_w);
                    let mut w = BandWorker::<T>::with_scratch(&shared_w.0, &shared_w.1, sc_w);
                    pipe_w.work(&mut w, &shared_w.2, false, None);
                    pipe_w.return_scratch(w.into_scratch());
                }));
            }));
            pws.push(pw);
        }

        let r_pw = pop_persistent_worker();
        let pipe_r = pipe.clone();
        let shared_r = shared.clone();
        let r_sc_worker = self.band_scratches.pop().unwrap_or_default();
        let stub_cdf = self.stub_cdf.take();
        let r_slot_w = r_slot.clone();
        let handoff_state_w = handoff_state.clone();
        let handoff_slot_w = handoff_slot.clone();
        let recycle_tx_w = recycle_tx.clone();
        let active_guard_r = ActiveWorkerGuard(active_workers.clone());
        let _ = r_pw.tx.send(Box::new(move || {
            let _active = active_guard_r;
            let handoff_guard = HandoffGuard(handoff_state_w.clone());
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _abort = AbortOnPanic(&pipe_r);
                let f_w = FrameCtx::new(&shared_r.0, &shared_r.1);
                let bp_w = &shared_r.2;
                let dsp_w = crate::decoder::av1::dsp::Dsp::<T>::new(f_w.bd);
                let mut cursor = stage::LogCursor::default();
                let mut r_w = BandWorker::<T>::with_scratch(&shared_r.0, &shared_r.1, r_sc_worker);
                let mut stub_ts = TileState::stub_with_cdf((0, f_w.mi_rows), (c0, c1), stub_cdf);
                let mut r_state = FrameState::<T> {
                    planes: r_planes,
                    above_line: r_above_line,
                    ipred: [IntraSetup::new(&f_w, 0), IntraSetup::new(&f_w, 1)],
                    edges: Default::default(),
                    cfl_ac: [0; crate::decoder::av1::dsp::ipred::MAX_CFL_AC],
                    ..Default::default()
                };
                let mut handed_off = false;
                let r_res = 'recon: {
                    loop {
                        if pipe_r.is_aborted() {
                            break;
                        }
                        if handoff_state_w.load(std::sync::atomic::Ordering::Acquire) == 1 {
                            match rx.recv() {
                                Ok(first) => {
                                    *handoff_slot_w
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                        Some(ReconHandoff {
                                            first,
                                            rx,
                                            cursor,
                                            stub_ts,
                                            r_state,
                                            r_sc,
                                        });
                                    handoff_state_w.store(2, std::sync::atomic::Ordering::Release);
                                    handed_off = true;
                                    break 'recon Ok(None);
                                }
                                Err(_) => break,
                            }
                        }
                        let msg = match rx.try_recv() {
                            Ok(m) => m,
                            Err(std::sync::mpsc::TryRecvError::Empty) => {
                                pipe_r.work(&mut r_w, bp_w, true, None);
                                if handoff_state_w.load(std::sync::atomic::Ordering::Acquire) == 1 {
                                    continue;
                                }
                                match rx.recv() {
                                    Ok(m) => m,
                                    Err(_) => break,
                                }
                            }
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                        };
                        if let Err(e) = step_bands_recon_msg(
                            msg,
                            &mut cursor,
                            &f_w,
                            &mut stub_ts,
                            &mut r_state,
                            &mut r_sc,
                            &dsp_w,
                            sb4,
                            sb_px,
                            sb_rows,
                            bp_w,
                            &pipe_r,
                            &mut r_w,
                            workers,
                            &recycle_tx_w,
                        ) {
                            break 'recon Err(e);
                        }
                    }
                    Ok(Some((
                        r_state.planes,
                        r_state.above_line,
                        r_sc,
                        stub_ts.cdf,
                    )))
                };
                match r_res {
                    Ok(None) => {
                        debug_assert!(handed_off);
                        drop(handoff_guard);
                        pipe_r.work(&mut r_w, bp_w, false, None);
                    }
                    Ok(Some(done_state)) => {
                        *r_slot_w
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(Ok(done_state));
                        drop(handoff_guard);
                        pipe_r.close();
                        pipe_r.work(&mut r_w, bp_w, false, None);
                    }
                    Err(e) => {
                        pipe_r.abort();
                        *r_slot_w
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Err(e));
                        drop(handoff_guard);
                    }
                }
                pipe_r.return_scratch(r_w.into_scratch());
            }));
        }));
        pws.push(r_pw);

        let main_sc = self.band_scratches.pop().unwrap_or_default();
        let res = (|| -> Result<()> {
            let _guard = AbortOnPanic(&pipe);
            let mut ts = TileState::with_cdf(
                fd.single_tile,
                fh,
                self.f.seq,
                (ti.mi_row_starts[0], ti.mi_row_starts[1]),
                (c0, c1),
                self.cdfs.pop(),
            );
            let mut next_chunk = || {
                recycle_rx
                    .try_recv()
                    .ok()
                    .or_else(|| self.pipe_chunks_free.pop())
                    .unwrap_or_default()
            };
            let mut current_chunk = next_chunk();
            let mut b = block::Block::default();
            let p_res = 'dec: {
                for r in 0..sb_rows {
                    let mi_row = r * sb4;
                    self.s.mi.slide_to(mi_row.saturating_sub(sb4));
                    self.s.left.clear();
                    let sb_y = r * sb_px;
                    self.s.set_sb_y(&self.f, sb_y);
                    {
                        let mut dec = SbDecoder {
                            f: &self.f,
                            t: &mut ts,
                            s: self.s,
                            sc: self.sc,
                            dsp: &self.dsp,
                            b,
                            recon: false,
                            sb_y,
                            read_deltas: false,
                            plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
                            stage: stage::DecoderStage::Parse(&mut current_chunk),
                            filter_type: [false; 2],
                        };
                        let mut c = c0;
                        while c < c1 {
                            if let Err(e) = dec.decode_sb(mi_row, c) {
                                break 'dec Err(e);
                            }
                            if dec.t.msac.overrun() {
                                break 'dec Err(Error::Truncated);
                            }
                            c += sb4;
                            if let stage::DecoderStage::Parse(ref mut chunk) = dec.stage {
                                if chunk.approx_size() >= LOG_CHUNK_FLUSH_BYTES {
                                    if tx
                                        .send(stage::PipeMsg::Chunk(std::mem::take(*chunk)))
                                        .is_err()
                                    {
                                        break 'dec Err(Error::Invalid("pipeline receiver closed"));
                                    }
                                    **chunk = next_chunk();
                                }
                            }
                        }
                        b = dec.b;
                    }
                    let tail_chunk = if !current_chunk.is_empty() {
                        let chunk = std::mem::take(&mut current_chunk);
                        current_chunk = next_chunk();
                        Some(chunk)
                    } else {
                        None
                    };
                    let mut bb = pipe.take_bufs();
                    if bp.deblock || bp.cdef {
                        let (m0, m1) = band_mi_rows(&self.f, r);
                        reset_band_mi(&mut bb.mi, &self.s.mi, &self.f, r);
                        copy_mi_rows(
                            &mut bb.mi,
                            &self.s.mi,
                            m0,
                            m1,
                            0,
                            self.f.mi_cols,
                            self.f.ss_x,
                            self.f.ss_y,
                        );
                    }
                    fill_band_meta(
                        &self.f,
                        &bp,
                        &mut bb,
                        r,
                        r + 1 == sb_rows,
                        &self.s.cdef_idx,
                        &self.s.lr,
                    );
                    if tx
                        .send(stage::PipeMsg::RowEnd {
                            r,
                            last: r + 1 == sb_rows,
                            bufs: Some(bb),
                            chunk: tail_chunk,
                        })
                        .is_err()
                    {
                        break 'dec Err(Error::Invalid("pipeline receiver closed"));
                    }
                }
                Ok(())
            };
            self.cdfs.push(ts.cdf);
            drop(tx);
            if !current_chunk.is_empty() {
                current_chunk.clear();
            }
            if self.pipe_chunks_free.len() < 16 && current_chunk.sbs.capacity() > 0 {
                self.pipe_chunks_free.push(current_chunk);
            }

            let mut p_w = BandWorker::<T>::with_scratch(&fd.seq, &fd.frame, main_sc);
            let mut took_over: Option<ReconOut<T>> = None;
            if p_res.is_err() {
                pipe.abort();
                while active_workers.load(std::sync::atomic::Ordering::Acquire) != 0 {
                    for _ in 0..64 {
                        std::hint::spin_loop();
                    }
                }
            } else {
                let _ = handoff_state.compare_exchange(
                    0,
                    1,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                );
                while handoff_state.load(std::sync::atomic::Ordering::Acquire) < 2 {
                    match pipe.try_next() {
                        SpinPoll::Task(t) => pipe.run(&mut p_w, &bp, t, None),
                        SpinPoll::Done => break,
                        SpinPoll::Idle | SpinPoll::Contended => {
                            for _ in 0..16 {
                                std::hint::spin_loop();
                            }
                        }
                    }
                }
                if handoff_state.load(std::sync::atomic::Ordering::Acquire) == 2 {
                    let ho = handoff_slot
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take();
                    if let Some(mut ho) = ho {
                        let takeover_res = (|| -> Result<()> {
                            step_bands_recon_msg(
                                ho.first,
                                &mut ho.cursor,
                                &self.f,
                                &mut ho.stub_ts,
                                &mut ho.r_state,
                                &mut ho.r_sc,
                                &self.dsp,
                                sb4,
                                sb_px,
                                sb_rows,
                                &bp,
                                &pipe,
                                &mut p_w,
                                workers + 1,
                                &recycle_tx,
                            )?;
                            while let Ok(msg) = ho.rx.recv() {
                                if pipe.is_aborted() {
                                    break;
                                }
                                step_bands_recon_msg(
                                    msg,
                                    &mut ho.cursor,
                                    &self.f,
                                    &mut ho.stub_ts,
                                    &mut ho.r_state,
                                    &mut ho.r_sc,
                                    &self.dsp,
                                    sb4,
                                    sb_px,
                                    sb_rows,
                                    &bp,
                                    &pipe,
                                    &mut p_w,
                                    workers + 1,
                                    &recycle_tx,
                                )?;
                            }
                            Ok(())
                        })();
                        match takeover_res {
                            Ok(()) => {
                                pipe.close();
                                took_over = Some(Ok((
                                    ho.r_state.planes,
                                    ho.r_state.above_line,
                                    ho.r_sc,
                                    ho.stub_ts.cdf,
                                )));
                            }
                            Err(e) => {
                                pipe.abort();
                                took_over = Some(Err(e));
                            }
                        }
                    } else {
                        pipe.abort();
                        took_over = Some(Err(Error::Invalid("missing handoff state")));
                    }
                }
                if took_over.as_ref().is_none_or(|r| r.is_ok()) {
                    pipe.work_spin(&mut p_w, &bp, None, || {
                        active_workers.load(std::sync::atomic::Ordering::Acquire) == 0
                    });
                } else {
                    while active_workers.load(std::sync::atomic::Ordering::Acquire) != 0 {
                        for _ in 0..64 {
                            std::hint::spin_loop();
                        }
                    }
                }
            }
            pipe.return_scratch(p_w.into_scratch());
            for pw in pws {
                push_persistent_worker(pw);
            }
            let r_res = match took_over {
                Some(r) => r,
                None => r_slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .ok_or(Error::Invalid("pipeline worker panicked"))?,
            };
            pipe.close();
            if p_res.is_err() {
                if let Err(r_err) = r_res {
                    return Err(r_err);
                }
                p_res?;
            }
            let (planes, above_line, sc, returned_cdf) = r_res?;
            if pipe.is_aborted() {
                return Err(Error::Invalid("pipeline worker panicked"));
            }
            *self.stub_cdf = Some(returned_cdf);
            self.s.planes = planes;
            self.s.above_line = above_line;
            *self.recon_sc = sc;
            Ok(())
        })();
        let out_planes = pipe.reclaim_into(
            self.band_free,
            self.band_carry,
            self.lf_grid,
            self.lr_free,
            self.band_scratches,
        );

        drop(recycle_tx);
        while let Ok(chunk) = recycle_rx.try_recv() {
            if self.pipe_chunks_free.len() < 16 {
                self.pipe_chunks_free.push(chunk);
            }
        }

        res?;
        self.finish_bands(out_planes, sink)
    }

    fn run_direct_pipe(&mut self, fd: &FrameData, sink: &mut dyn RowSink<T>) -> Result<()> {
        let f = &self.f;
        let ti = &f.fh.tile_info;
        let sb4 = f.sb4;
        let sb_px = f.sb_px;
        let sb_rows = f.sb_rows;
        let fh = f.fh;

        self.post_start = 0;
        self.row_end = sb_rows;
        self.col_range = (0, f.sb_cols * sb4);
        self.next_stripe = 0;
        self.next_bound = 1;
        self.rows_end = fh.frame_height;

        let c0 = ti.mi_col_starts[0];
        let c1 = ti.mi_col_starts[1];

        // Unbounded channel so Stage P on the main thread never sleeps on
        // backpressure (sleeping decays PELT utilization on heterogeneous
        // ARM big.LITTLE cores and causes wakeup migration off the Prime
        // core while mid cores are frequency-capped below misfit).
        let (tx, rx) = std::sync::mpsc::channel::<stage::PipeMsg<T>>();
        let (recycle_tx, recycle_rx) = std::sync::mpsc::channel::<stage::LogChunk>();

        let mut direct_bufs = std::mem::take(&mut self.direct_bufs);
        let mut r_sc = std::mem::take(self.recon_sc);
        let np = f.num_planes;
        let mi_cols_al = f.sb_cols * f.sb4;
        let mi_rows_al = f.sb_rows * f.sb4;
        let win_rows_luma = mi_rows_al * 4;

        let stub_cdf = self.stub_cdf.take();
        type DirectDone<T> = ([PlaneWin<T>; 3], Scratch, Box<CdfContext>);
        type DirectOut<T> = Result<Option<DirectDone<T>>>;
        let out_slot: std::sync::Arc<Mutex<Option<DirectOut<T>>>> =
            std::sync::Arc::new(Mutex::new(None));
        let handoff_state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
        let handoff_slot: std::sync::Arc<Mutex<Option<ReconHandoff<T>>>> =
            std::sync::Arc::new(Mutex::new(None));
        let done_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let out_slot_w = out_slot.clone();
        let handoff_state_w = handoff_state.clone();
        let handoff_slot_w = handoff_slot.clone();
        let done_guard = WorkerDoneGuard(done_flag.clone());
        let seq_owned = f.seq.clone();
        let fh_owned = f.fh.clone();
        let recycle_tx_w = recycle_tx.clone();
        let worker = pop_persistent_worker();
        let _ = worker.tx.send(Box::new(move || {
            let _done = done_guard;
            let _handoff = HandoffGuard(handoff_state_w.clone());
            if let Ok(r) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || -> DirectOut<T> {
                    let f_w = FrameCtx::new(&seq_owned, &fh_owned);
                    let dsp_w = crate::decoder::av1::dsp::Dsp::<T>::new(f_w.bd);
                    let mut cursor = stage::LogCursor::default();
                    let mut stub_ts =
                        TileState::stub_with_cdf((0, f_w.mi_rows), (c0, c1), stub_cdf);
                    let mut r_planes: [PlaneWin<T>; 3] = Default::default();
                    for p in 0..np {
                        let sy = f_w.sub(p).1;
                        let stride = ((mi_cols_al * 4) >> f_w.sub(p).0).next_multiple_of(64);
                        let buf = std::mem::take(&mut direct_bufs[p]);
                        r_planes[p] = PlaneWin::from_vec(buf, stride, win_rows_luma >> sy);
                    }
                    let mut r_state = FrameState::<T> {
                        planes: r_planes,
                        above_line: Default::default(),
                        ipred: [IntraSetup::new(&f_w, 0), IntraSetup::new(&f_w, 1)],
                        edges: Default::default(),
                        cfl_ac: [0; crate::decoder::av1::dsp::ipred::MAX_CFL_AC],
                        ..Default::default()
                    };
                    loop {
                        if handoff_state_w.load(std::sync::atomic::Ordering::Acquire) == 1 {
                            match rx.recv() {
                                Ok(first) => {
                                    *handoff_slot_w
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                        Some(ReconHandoff {
                                            first,
                                            rx,
                                            cursor,
                                            stub_ts,
                                            r_state,
                                            r_sc,
                                        });
                                    handoff_state_w.store(2, std::sync::atomic::Ordering::Release);
                                    return Ok(None);
                                }
                                Err(_) => break,
                            }
                        }
                        let Ok(msg) = rx.recv() else { break };
                        if let stage::PipeMsg::Chunk(mut chunk) = msg {
                            decode_log_chunk(
                                &chunk,
                                &mut cursor,
                                &f_w,
                                &mut stub_ts,
                                &mut r_state,
                                &mut r_sc,
                                &dsp_w,
                                sb4,
                            )?;
                            chunk.clear();
                            let _ = recycle_tx_w.send(chunk);
                        }
                    }
                    Ok(Some((r_state.planes, r_sc, stub_ts.cdf)))
                }))
            {
                *out_slot_w
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
            }
        }));

        let mut ts = TileState::with_cdf(
            fd.single_tile,
            fh,
            f.seq,
            (ti.mi_row_starts[0], ti.mi_row_starts[1]),
            (c0, c1),
            self.cdfs.pop(),
        );
        let mut next_chunk = || {
            recycle_rx
                .try_recv()
                .ok()
                .or_else(|| self.pipe_chunks_free.pop())
                .unwrap_or_default()
        };
        let mut current_chunk = next_chunk();
        let mut b = block::Block::default();
        let p_res = 'dec: {
            for r in 0..sb_rows {
                let mi_row = r * sb4;
                self.s.mi.slide_to(mi_row.saturating_sub(sb4));
                self.s.left.clear();
                let sb_y = r * sb_px;
                self.s.set_sb_y(&self.f, sb_y);
                {
                    let mut dec = SbDecoder {
                        f: &self.f,
                        t: &mut ts,
                        s: self.s,
                        sc: self.sc,
                        dsp: &self.dsp,
                        b,
                        recon: false,
                        sb_y,
                        read_deltas: false,
                        plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
                        stage: stage::DecoderStage::Parse(&mut current_chunk),
                        filter_type: [false; 2],
                    };
                    let mut c = c0;
                    while c < c1 {
                        if let Err(e) = dec.decode_sb(mi_row, c) {
                            break 'dec Err(e);
                        }
                        if dec.t.msac.overrun() {
                            break 'dec Err(Error::Truncated);
                        }
                        c += sb4;
                        if let stage::DecoderStage::Parse(ref mut chunk) = dec.stage {
                            if chunk.approx_size() >= LOG_CHUNK_FLUSH_BYTES {
                                if tx
                                    .send(stage::PipeMsg::Chunk(std::mem::take(*chunk)))
                                    .is_err()
                                {
                                    break 'dec Err(Error::Invalid("pipeline receiver closed"));
                                }
                                **chunk = next_chunk();
                            }
                        }
                    }
                    b = dec.b;
                }
            }
            if !current_chunk.is_empty() {
                let chunk = std::mem::take(&mut current_chunk);
                if tx.send(stage::PipeMsg::Chunk(chunk)).is_err() {
                    break 'dec Err(Error::Invalid("pipeline receiver closed"));
                }
            } else if self.pipe_chunks_free.len() < 16 && current_chunk.sbs.capacity() > 0 {
                self.pipe_chunks_free.push(current_chunk);
            }
            Ok(())
        };
        self.cdfs.push(ts.cdf);
        drop(tx);

        let mut took_over: Option<Result<DirectDone<T>>> = None;
        if p_res.is_ok() {
            let _ = handoff_state.compare_exchange(
                0,
                1,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            );
            while handoff_state.load(std::sync::atomic::Ordering::Acquire) < 2 {
                std::hint::spin_loop();
            }
            if handoff_state.load(std::sync::atomic::Ordering::Acquire) == 2 {
                let ho = handoff_slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if let Some(mut ho) = ho {
                    let takeover_res = (|| -> Result<DirectDone<T>> {
                        let mut step = |msg: stage::PipeMsg<T>| -> Result<()> {
                            if let stage::PipeMsg::Chunk(mut chunk) = msg {
                                decode_log_chunk(
                                    &chunk,
                                    &mut ho.cursor,
                                    &self.f,
                                    &mut ho.stub_ts,
                                    &mut ho.r_state,
                                    &mut ho.r_sc,
                                    &self.dsp,
                                    sb4,
                                )?;
                                chunk.clear();
                                if self.pipe_chunks_free.len() < 16 {
                                    self.pipe_chunks_free.push(chunk);
                                }
                            }
                            Ok(())
                        };
                        step(ho.first)?;
                        while let Ok(msg) = ho.rx.recv() {
                            step(msg)?;
                        }
                        Ok((ho.r_state.planes, ho.r_sc, ho.stub_ts.cdf))
                    })();
                    took_over = Some(takeover_res);
                } else {
                    took_over = Some(Err(Error::Invalid("missing handoff state")));
                }
            }
        }

        while !done_flag.load(std::sync::atomic::Ordering::Acquire) {
            std::hint::spin_loop();
        }
        drop(recycle_tx);
        while let Ok(chunk) = recycle_rx.try_recv() {
            if self.pipe_chunks_free.len() < 16 {
                self.pipe_chunks_free.push(chunk);
            }
        }
        let r_opt = out_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if r_opt.is_some() {
            push_persistent_worker(worker);
        }
        let r_res: Result<DirectDone<T>> = match took_over {
            Some(r) => r,
            None => match r_opt {
                Some(Ok(Some(done))) => Ok(done),
                Some(Ok(None)) => Err(Error::Invalid("missing direct pipeline output")),
                Some(Err(e)) => Err(e),
                None => Err(Error::Invalid("pipeline worker panicked")),
            },
        };
        if p_res.is_err() {
            if let Err(r_err) = r_res {
                return Err(r_err);
            }
            p_res?;
        }
        let (planes, sc, returned_cdf) = r_res?;
        *self.stub_cdf = Some(returned_cdf);
        self.s.planes = planes;
        *self.recon_sc = sc;
        if self.direct {
            self.finish_direct(sink)?;
        }
        Ok(())
    }

    /// Multi-tile frames with post filters (whole frame, output planes
    /// handed to the sink): the band pipeline of [`Pipeline::run_bands`]
    /// with the rows produced by all threads. The decoding units are the
    /// superblock rows of the tiles in raster order ([`TileJobs`]): each is
    /// decoded with the persistent state of its tile and copied
    /// (pixels, mode info, `cdef_idx`, restoration units) into the row
    /// buffer of its superblock row; a row is queued for the finishing
    /// stage once all its tiles are in.
    fn run_tiles_bands(
        &mut self,
        fd: &FrameData,
        threads: usize,
        sink: &mut dyn RowSink<T>,
    ) -> Result<()> {
        let (bp, out_planes) = self.band_setup();
        let f = &self.f;
        let ti = &f.fh.tile_info;
        let tile_row = (0..f.sb_rows)
            .map(|r| {
                (0..ti.rows)
                    .find(|&t| r * f.sb4 < ti.mi_row_starts[t + 1])
                    .unwrap_or(ti.rows - 1)
            })
            .collect();
        let ctxs = (0..ti.rows * ti.cols)
            .map(|_| {
                Mutex::new(TileCtx::from_worker(
                    self.tile_workers.pop().unwrap_or_else(TileWorker::new),
                ))
            })
            .collect();
        let jobs = TileJobs {
            fd,
            ctxs,
            slots: (0..f.sb_rows)
                .map(|_| {
                    Mutex::new(RowSlot {
                        bufs: None,
                        remaining: ti.cols,
                    })
                })
                .collect(),
            order: Mutex::new(RowOrder {
                next: 0,
                ready: (0..f.sb_rows).map(|_| None).collect(),
                carry: std::mem::take(self.order_carry),
                above_ready: (0..f.sb_rows).map(|r| r == 0).collect(),
                carries: (0..f.sb_rows).map(|_| None).collect(),
                free_carries: Vec::with_capacity(4),
            }),
            meta: Mutex::new(TileMeta {
                cdef_idx: std::mem::take(&mut self.s.cdef_idx),
                lr: std::mem::take(&mut self.s.lr),
            }),
            err: Mutex::new(None),
            tile_row,
            mi_arrays: self.mi_arrays,
        };
        let (aborted, out_planes) = {
            let pipe = BandPipeline::with_pooled(
                out_planes,
                std::mem::take(self.band_free),
                std::mem::take(self.band_carry),
                std::mem::take(self.lf_grid),
                std::mem::take(self.lr_free),
                threads,
            );
            pipe.set_parallel_bands(bp.deblock || bp.cdef || bp.lr);
            // A tile's rows are decoded in order; tiles (also of different
            // tile rows) are independent.
            let lanes = (0..ti.rows)
                .flat_map(|tr| {
                    let (r0, r1) = (
                        ti.mi_row_starts[tr] / f.sb4,
                        ti.mi_row_starts[tr + 1].div_ceil(f.sb4),
                    );
                    (0..ti.cols).map(move |tc| (r0 * ti.cols + tc, r1 * ti.cols + tc))
                })
                .collect();
            pipe.set_units(lanes, ti.cols, threads);
            let mut worker_scratches: Vec<BandWorkerScratch<T>> = (1..threads)
                .map(|_| self.band_scratches.pop().unwrap_or_default())
                .collect();
            let main_sc = self.band_scratches.pop().unwrap_or_default();
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(threads.saturating_sub(1));
                for sc in worker_scratches.drain(..) {
                    let (pipe, bp, jobs) = (&pipe, &bp, &jobs);
                    handles.push(scope.spawn(move || {
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                            let _abort = AbortOnPanic(pipe);
                            let mut w = BandWorker::<T>::with_scratch(&fd.seq, &fd.frame, sc);
                            pipe.work(&mut w, bp, false, Some(jobs));
                            pipe.return_scratch(w.into_scratch());
                        }));
                    }));
                }
                let mut main_w = BandWorker::<T>::with_scratch(&fd.seq, &fd.frame, main_sc);
                pipe.work_spin(&mut main_w, &bp, Some(&jobs), || {
                    handles.iter().all(|h| h.is_finished())
                });
                pipe.return_scratch(main_w.into_scratch());
                drop(handles);
            });
            let aborted = pipe.is_aborted();
            let out_planes = pipe.reclaim_into(
                self.band_free,
                self.band_carry,
                self.lf_grid,
                self.lr_free,
                self.band_scratches,
            );
            (aborted, out_planes)
        };
        let TileJobs {
            ctxs,
            slots,
            order,
            meta,
            err,
            ..
        } = jobs;
        for ctx in ctxs {
            self.tile_workers.push(
                ctx.into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .into_worker(),
            );
        }
        for slot in slots {
            if let Some(bb) = slot
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .bufs
            {
                self.band_free.push(bb);
            }
        }
        let order = order
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for bb in order.ready.into_iter().flatten() {
            self.band_free.push(bb);
        }
        for c in order
            .free_carries
            .into_iter()
            .chain(order.carries.into_iter().flatten())
        {
            if let Some(bb) = self
                .band_free
                .iter_mut()
                .find(|bb| bb.carry_planes[0].stride == 0)
            {
                bb.carry_planes = c.planes;
                bb.carry_mi = c.mi;
            }
        }
        *self.order_carry = order.carry;
        let meta = meta
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (self.s.cdef_idx, self.s.lr) = (meta.cdef_idx, meta.lr);
        if let Some((_, e)) = err
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Err(e);
        }
        if aborted {
            return Err(Error::Invalid("decoding thread panicked"));
        }
        self.finish_bands(out_planes, sink)
    }

    /// Frame parameters and output planes of [`Pipeline::run_bands`] /
    /// [`Pipeline::run_tiles_bands`] (whole frame).
    fn band_setup(&mut self) -> (BandParams, [PlaneWin<T>; 3]) {
        let f = &self.f;
        let fh = f.fh;
        self.post_start = 0;
        self.row_end = f.sb_rows;
        self.col_range = (0, f.sb_cols * f.sb4);
        self.rows_end = fh.frame_height;
        let mut bp = BandParams {
            deblock: self.deblock,
            cdef: self.cdef && cdef_active(fh),
            lr: self.lr,
            lr_on: [false; 3],
            mi_w: self.mi_w,
            mi_h: self.mi_h,
            out_w: self.out_w,
            out_h: self.out_h,
            out_stride: [0; 3],
            cdef_stride: self.s.cdef_stride,
            frame_h: fh.frame_height,
            col_range: self.col_range,
        };
        let mut out_planes: [PlaneWin<T>; 3] = Default::default();
        let mut bufs = pop_direct_planes::<T>(f.num_planes);
        for p in 0..f.num_planes {
            bp.lr_on[p] = self.lr && fh.lr.frame_restoration_type[p] != RestorationType::None;
            bp.out_stride[p] = self.s.planes[p].stride;
            out_planes[p] = PlaneWin::from_vec_uninit(
                std::mem::take(&mut bufs[p]),
                self.out_w[p],
                self.out_h[p],
            );
        }
        (bp, out_planes)
    }

    /// Hands the output planes of the band pipeline over to the sink.
    fn finish_bands(
        &mut self,
        mut out_planes: [PlaneWin<T>; 3],
        sink: &mut dyn RowSink<T>,
    ) -> Result<()> {
        let planes = (0..self.f.num_planes)
            .map(|p| {
                let w = std::mem::take(&mut out_planes[p]);
                let stride = w.stride;
                let (mut data, offset) = w.into_raw();
                let need = stride * self.out_h[p];
                if data.len() < need {
                    data.resize(need, T::default());
                }
                crate::decoder::av1::sink::Plane {
                    width: self.out_w[p],
                    height: self.out_h[p],
                    stride,
                    offset,
                    data,
                }
            })
            .collect();
        sink.finish(planes)
    }

    /// The [`RowJob`] of superblock row `r` (see [`Pipeline::run_bands`]),
    /// called after the row has been decoded: the row buffer in
    /// `self.s.planes` moves into the job (the next row is decoded into
    /// `bb.planes`), with copies of the mode info rows deblocking and CDEF
    /// read, the `cdef_idx` rows of the band and the restoration unit rows
    /// of its stripes.
    fn capture_row(
        &mut self,
        r: usize,
        last: bool,
        bp: &BandParams,
        mut bb: Box<BandBufs<T>>,
    ) -> RowJob<T> {
        let f = &self.f;
        let sb_px = f.sb_px;
        for p in 0..f.num_planes {
            let sy = f.sub(p).1;
            if !last {
                let stride = self.s.planes[p].stride;
                bb.planes[p].reset(
                    stride,
                    (sb_px + BAND_TOP) >> sy,
                    0,
                    ((r + 1) * sb_px - BAND_TOP) >> sy,
                );
            }
            std::mem::swap(&mut self.s.planes[p], &mut bb.planes[p]);
        }
        if bp.deblock || bp.cdef {
            let (m0, m1) = band_mi_rows(f, r);
            reset_band_mi(&mut bb.mi, &self.s.mi, f, r);
            copy_mi_rows(&mut bb.mi, &self.s.mi, m0, m1, 0, f.mi_cols, f.ss_x, f.ss_y);
        }
        fill_band_meta(f, bp, &mut bb, r, last, &self.s.cdef_idx, &self.s.lr);
        RowJob { r, last, bufs: bb }
    }

    /// Hands the frame planes over to the sink (direct mode).
    fn finish_direct(&mut self, sink: &mut dyn RowSink<T>) -> Result<()> {
        let f = &self.f;
        let planes = (0..f.num_planes)
            .map(|p| {
                let lr_p = self.lr
                    && !self.intrabc
                    && f.fh.lr.frame_restoration_type[p] != RestorationType::None;
                let w = if lr_p {
                    std::mem::take(&mut self.lr_out_planes[p])
                } else {
                    std::mem::take(&mut self.s.planes[p])
                };
                debug_assert!(w.y0 == 0 && w.rows >= self.out_h[p] && w.stride >= self.out_w[p]);
                let stride = w.stride;
                let (mut data, offset) = w.into_raw();
                let need = stride * self.out_h[p];
                if data.len() < need {
                    data.resize(need, T::default());
                }
                crate::decoder::av1::sink::Plane {
                    width: self.out_w[p],
                    height: self.out_h[p],
                    stride,
                    offset,
                    data,
                }
            })
            .collect();
        sink.finish(planes)
    }

    /// Multi-tile frames without post filters (whole frame, direct
    /// output): the tiles are decoded in parallel, each worker into a
    /// window of one superblock row of its tile, and every decoded
    /// superblock row is copied into `self.s.planes` right away (while
    /// still in the cache) under the lock of that frame superblock row.
    fn run_tiles_direct(
        &mut self,
        fd: &FrameData,
        threads: usize,
        sink: &mut dyn RowSink<T>,
    ) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let f = &self.f;
        let ti = &f.fh.tile_info;
        let sb_px = f.sb_px;
        let num_jobs = ti.rows * ti.cols;
        let num_workers = threads.min(num_jobs);
        let strides = std::array::from_fn(|p| self.s.planes[p].stride);
        let [p0, p1, p2] = &mut self.s.planes;
        let mut c0 = p0.data_mut().chunks_mut((sb_px >> f.sub(0).1) * strides[0]);
        let mut c1 = p1
            .data_mut()
            .chunks_mut(((sb_px >> f.sub(1).1) * strides[1]).max(1));
        let mut c2 = p2
            .data_mut()
            .chunks_mut(((sb_px >> f.sub(2).1) * strides[2]).max(1));
        let chunks: Vec<Mutex<[&mut [T]; 3]>> = (0..f.sb_rows)
            .map(|_| {
                Mutex::new([
                    c0.next().unwrap_or(&mut []),
                    c1.next().unwrap_or(&mut []),
                    c2.next().unwrap_or(&mut []),
                ])
            })
            .collect();
        let dsp = &self.dsp;
        let mi_h = &self.mi_h;
        // Largest tiles (by coded size, a proxy for decoding time) first so a
        // heavy tile is not started near the end while the other threads idle.
        let mut order: Vec<usize> = (0..num_jobs).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(fd.tiles()[i].len()));
        let next = AtomicUsize::new(0);
        let min_failed_key = AtomicUsize::new(usize::MAX);
        let mut workers: Vec<TileWorker<T>> = (0..num_workers)
            .map(|_| {
                let mut w = self.tile_workers.pop().unwrap_or_else(TileWorker::new);
                if w.cdf.is_none() {
                    w.cdf = self.cdfs.pop();
                }
                w
            })
            .collect();
        let work = |w: &mut TileWorker<T>| -> Option<(usize, Error)> {
            let mut err: Option<(usize, Error)> = None;
            loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= num_jobs {
                    break;
                }
                let idx = order[j];
                let (tr, tc) = (idx / ti.cols, idx % ti.cols);
                if let Err((key, e)) = w.decode_tile(
                    f,
                    dsp,
                    fd.tiles()[idx],
                    tr,
                    tc,
                    &chunks,
                    &strides,
                    mi_h,
                    &min_failed_key,
                ) {
                    min_failed_key.fetch_min(key, Ordering::Relaxed);
                    if err.as_ref().is_none_or(|(k, _)| key < *k) {
                        err = Some((key, e));
                    }
                }
            }
            err
        };
        let mut slots: Vec<Option<Option<(usize, Error)>>> =
            (0..workers.len().saturating_sub(1)).map(|_| None).collect();
        let mut main_err: Option<(usize, Error)> = None;
        std::thread::scope(|s| {
            let Some((first, rest)) = workers.split_first_mut() else {
                main_err = Some((0, Error::Invalid("no tile workers")));
                return;
            };
            let handles: Vec<_> = rest
                .iter_mut()
                .zip(slots.iter_mut())
                .map(|(w, slot)| {
                    s.spawn(|| {
                        if let Ok(r) =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(w)))
                        {
                            *slot = Some(r);
                        }
                    })
                })
                .collect();
            main_err = work(first);
            while handles.iter().any(|h| !h.is_finished()) {
                for _ in 0..64 {
                    std::hint::spin_loop();
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            drop(handles);
        });
        let mut errs = Vec::with_capacity(workers.len().max(1));
        errs.push(main_err);
        errs.extend(slots.into_iter().map(|slot| {
            slot.unwrap_or_else(|| Some((0, Error::Invalid("decoding thread panicked"))))
        }));
        self.tile_workers.extend(workers);
        if let Some((_, e)) = errs.into_iter().flatten().min_by_key(|(k, _)| *k) {
            return Err(e);
        }
        self.finish_direct(sink)
    }

    /// Single-threaded counterpart of [`Pipeline::run_tiles_direct`]: decodes
    /// each tile from top to bottom directly into `self.s.planes` using a
    /// single pooled `CdfContext` and tile-width `MiWin`/`AboveCtx`.
    fn run_tiles_direct_st(&mut self, fd: &FrameData, sink: &mut dyn RowSink<T>) -> Result<()> {
        let f = &self.f;
        let fh = f.fh;
        let ti = &fh.tile_info;
        let np = f.num_planes;
        let (sb4, sb_px) = (f.sb4, f.sb_px);
        for p in 0..np {
            self.s.above_line[p].clear();
        }
        self.s.ipred = [IntraSetup::new(f, 0), IntraSetup::new(f, 1)];
        let mut cdf = self.cdfs.pop();
        let mut first_err: Option<(usize, Error)> = None;
        for tr in 0..ti.rows {
            let (mi_r0, mi_r1) = (ti.mi_row_starts[tr], ti.mi_row_starts[tr + 1]);
            let (sb_r0, sb_r1) = (mi_r0 / sb4, mi_r1.div_ceil(sb4));
            for tc in 0..ti.cols {
                let (mi_c0, mi_c1) = (ti.mi_col_starts[tc], ti.mi_col_starts[tc + 1]);
                let tile_mi_al = (mi_c1 - mi_c0).div_ceil(sb4) * sb4;
                self.s.mi.reset(
                    tile_mi_al + 32,
                    2 * sb4,
                    mi_c0,
                    (sb_r0 * sb4).saturating_sub(sb4),
                    f.ss_x,
                    f.ss_y,
                    false,
                    false,
                    false,
                    fh.allow_intrabc,
                );
                self.s
                    .above
                    .reset(mi_c0, tile_mi_al + 32, np, fh.allow_screen_content_tools);
                let data = fd.tiles()[tr * ti.cols + tc];
                let mut ts = TileState::with_cdf(
                    data,
                    fh,
                    self.f.seq,
                    (mi_r0, mi_r1),
                    (mi_c0, mi_c1),
                    cdf.take(),
                );
                for r in sb_r0..sb_r1 {
                    let key = (r << 16) | (tr << 8) | tc;
                    if first_err.as_ref().is_some_and(|(k, _)| key > *k) {
                        break;
                    }
                    let mi_row = r * sb4;
                    let sb_y = r * sb_px;
                    self.s.mi.slide_to(mi_row.saturating_sub(sb4));
                    self.s.left.clear();
                    let mut dec = SbDecoder {
                        f,
                        t: &mut ts,
                        s: &mut self.s,
                        sc: &mut self.sc,
                        dsp: &self.dsp,
                        b: Default::default(),
                        recon: true,
                        sb_y,
                        read_deltas: false,
                        plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
                        stage: stage::DecoderStage::Fused,
                        filter_type: [false; 2],
                    };
                    let mut c = mi_c0;
                    let mut row_err = None;
                    while c < mi_c1 {
                        if let Err(e) = dec.decode_sb(mi_row, c) {
                            row_err = Some(e);
                            break;
                        }
                        if dec.t.msac.overrun() {
                            row_err = Some(Error::Truncated);
                            break;
                        }
                        c += sb4;
                    }
                    if let Some(e) = row_err {
                        if first_err.as_ref().is_none_or(|(k, _)| key < *k) {
                            first_err = Some((key, e));
                        }
                        break;
                    }
                }
                cdf = Some(ts.cdf);
            }
        }
        if let Some(c) = cdf {
            self.cdfs.push(c);
        }
        if let Some((_, e)) = first_err {
            return Err(e);
        }
        self.finish_direct(sink)
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

        let mut panicked = false;
        type MtBatch<T> = Vec<(
            usize,
            usize,
            std::result::Result<FrameState<T>, (usize, Error)>,
        )>;
        let mut slots: Vec<Option<MtBatch<T>>> = (1..num_workers).map(|_| None).collect();
        let main_out: MtBatch<T> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_workers.saturating_sub(1));
            for slot in slots.iter_mut() {
                handles.push(s.spawn(|| {
                    if let Ok(out) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut out = Vec::new();
                        loop {
                            let idx = next_job.fetch_add(1, Ordering::Relaxed);
                            if idx >= num_jobs {
                                break;
                            }
                            let (tr, tc) = tile_jobs[idx];
                            let data = fd.tiles()[tr * ti.cols + tc];
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
                    })) {
                        *slot = Some(out);
                    }
                }));
            }
            let mut main_out = Vec::new();
            loop {
                let idx = next_job.fetch_add(1, Ordering::Relaxed);
                if idx >= num_jobs {
                    break;
                }
                let (tr, tc) = tile_jobs[idx];
                let data = fd.tiles()[tr * ti.cols + tc];
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
            while handles.iter().any(|h| !h.is_finished()) {
                for _ in 0..64 {
                    std::hint::spin_loop();
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            drop(handles);
            main_out
        });
        let mut worker_results = Vec::with_capacity(num_workers);
        worker_results.push(main_out);
        for slot in slots {
            match slot {
                Some(out) => worker_results.push(out),
                None => panicked = true,
            }
        }
        if panicked {
            return Err(Error::Invalid("decoding thread panicked"));
        }

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
            self.finish_direct(sink)?;
        }
        Ok(())
    }

    /// Decodes a single tile `(tr, tc)` as an independent AV1 unit into its own compact
    /// `FrameState<T>`.
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
        let mut planes: [PlaneWin<T>; 3] = Default::default();
        let win_rows_luma = if !recon { 0 } else { (retain_rows + 2) * sb_px + 64 };
        for p in 0..np {
            let (sx, sy) = f.sub(p);
            let x0 = (mi_c0 * 4) >> sx;
            let y0 = (sb_r0 * sb_px) >> sy;
            let stride = if recon { ((mi_cols_tile_al * 4) >> sx).next_multiple_of(64) } else { 0 };
            let rows = if recon { (win_rows_luma >> sy) + 16 } else { 0 };
            planes[p] = PlaneWin::new_tile(stride, rows, x0, y0);
        }

        let above_w = mi_cols_tile_al + 32;
        let mi_rows = if recon { ((retain_rows + 2) * sb4).next_power_of_two() } else { 2 * sb4 };
        let cdef_row0 = (sb_r0 * sb4) / 16;
        let cdef_col0 = mi_c0 / 16;
        let cdef_rows = (sb_r1.saturating_sub(sb_r0) * sb4) / 16 + 2;
        let cdef_stride = mi_cols_tile_al / 16 + 2;

        let mut above = AboveCtx::default();
        above.reset(mi_c0, above_w, np, fh.allow_screen_content_tools);
        let mut tile_s = FrameState {
            planes,
            // No deblocking runs while decoding the tile, so `planes[p]` holds
            // unfiltered rows directly and `above_line` is not needed.
            above_line: Default::default(),
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
            above,
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
            cfl_ac: [0; crate::decoder::av1::dsp::ipred::MAX_CFL_AC],
            edges: crate::decoder::av1::dsp::ipred::IntraEdges::default(),
            ipred: [IntraSetup::new(f, 0), IntraSetup::new(f, 1)],
        };

        let mut ts = TileState::new(data, fh, f.seq, (mi_r0, mi_r1), (mi_c0, mi_c1));
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
                stage: stage::DecoderStage::Fused,
                filter_type: [false; 2],
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
        self.save_above_line(r);
        let f = &self.f;
        if self.deblock && r >= self.post_start {
            deblock::deblock_rows(
                f,
                &self.s.mi,
                &mut self.s.planes,
                r * f.sb4,
                (r + 1) * f.sb4,
                self.col_range,
                &self.dsp,
                self.lf_grid,
            );
        }
        Ok(())
    }

    /// Saves the unfiltered bottom row of superblock row `r` for intra
    /// prediction of row r + 1 (if `above_line` is used).
    fn save_above_line(&mut self, r: usize) {
        let f = &self.f;
        if r + 1 < self.row_end && !self.s.above_line[0].is_empty() {
            for p in 0..f.num_planes {
                let sy = f.sub(p).1;
                let y = (((r + 1) * f.sb_px) >> sy) - 1;
                let y = y.min(self.s.planes[p].y0 + self.s.planes[p].rows - 1);
                let row = &self.s.planes[p].row(y)[..self.mi_w[p]];
                self.s.above_line[p][..row.len()].copy_from_slice(row);
            }
        }
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
        if self.lr && (!self.direct || self.cdef || self.superres || self.intrabc) {
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
                self.cdef_srcs,
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
        if !self.direct || self.lr || (!self.intrabc && sink.wants_rows_final()) {
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
            for p in (0..np).filter(|&p| lr_on(p)) {
                let g = &geoms[p];
                let above = self.lr_bounds.iter().find(|b| b.k == k).map(|b| &b.rows[p]);
                let below = self
                    .lr_bounds
                    .iter()
                    .find(|b| b.k == k + 1)
                    .map(|b| &b.rows[p]);
                if self.intrabc {
                    lr_apply::lr_stripe_in_place(
                        &self.s.lr[p],
                        g,
                        &mut self.s.planes[p],
                        above,
                        below,
                        &mut self.lr_scratch,
                        f.sub(p).1,
                        f.bd,
                        &self.dsp,
                    );
                } else {
                    let y_first = g.start_y.max(0) as usize;
                    let stride = self.lr_out_planes[p].stride;
                    let out = &mut self.lr_out_planes[p].data_mut()[y_first * stride..];
                    lr_apply::lr_stripe(
                        &self.s.lr[p],
                        g,
                        &self.s.planes[p],
                        above,
                        below,
                        out,
                        stride,
                        &mut self.lr_scratch,
                        f.sub(p).1,
                        f.bd,
                        (0, usize::MAX),
                        &self.dsp,
                    );
                }
            }
            // Intra block copy reads unfiltered samples of the whole frame.
            if !self.intrabc && sink.wants_rows_final() {
                let y_end = (64 * k + 56).min(geoms[0].plane_h);
                let mut strides = [0usize; 3];
                let (s0, s12) = self.s.planes.split_at_mut(1);
                let (s1, s2) = s12.split_at_mut(1);
                let (l0, l12) = self.lr_out_planes.split_at_mut(1);
                let (l1, l2) = l12.split_at_mut(1);
                let p0 = if lr_on(0) { &mut l0[0] } else { &mut s0[0] };
                let p1 = if lr_on(1) { &mut l1[0] } else { &mut s1[0] };
                let p2 = if lr_on(2) { &mut l2[0] } else { &mut s2[0] };
                strides[0] = p0.stride;
                strides[1] = p1.stride;
                strides[2] = p2.stride;
                let mut refs = [p0.data_mut(), p1.data_mut(), p2.data_mut()];
                sink.rows_final(&mut refs[..np], &strides[..np], y_end);
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
        let mut bands = [
            PlaneRows {
                y0: 0,
                h: 0,
                w: 0,
                data: &[],
                stride: 0,
            },
            PlaneRows {
                y0: 0,
                h: 0,
                w: 0,
                data: &[],
                stride: 0,
            },
            PlaneRows {
                y0: 0,
                h: 0,
                w: 0,
                data: &[],
                stride: 0,
            },
        ];
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
            bands[p] = PlaneRows {
                y0: y_first,
                h: y_last + 1 - y_first,
                w: self.out_w[p],
                data,
                stride,
            };
        }
        sink.push(&bands[..np])
    }
}

/// Whether CDEF changes any sample (some strength of a used preset is
/// non-zero).
fn cdef_active(fh: &crate::decoder::av1::headers::FrameHeader) -> bool {
    let cd = &fh.cdef;
    let n = (1usize << cd.bits).min(8);
    [
        &cd.y_pri_strength,
        &cd.y_sec_strength,
        &cd.uv_pri_strength,
        &cd.uv_sec_strength,
    ]
    .iter()
    .any(|s| s[..n].iter().any(|&v| v != 0))
}

/// Luma rows above superblock row `r` in its row buffer of
/// [`Pipeline::run_bands`] (copied from the row above before deblocking):
/// deblocking reads 7 rows above the row, the band of the row starts 8
/// rows above it and CDEF reads 2 more above that.
const BAND_TOP: usize = 16;

/// Luma rows `lum0..lum1` of the band of superblock row `r` in
/// [`Pipeline::run_bands`]: the rows that are final once the row has been
/// deblocked (all of it but its last [`POST_LAG`] rows, plus the last rows
/// of the row above, like `Pipeline::post_row`).
fn band_rows(r: usize, last: bool, sb_px: usize) -> (usize, usize) {
    let start = |r: usize| if r == 0 { 0 } else { r * sb_px - POST_LAG };
    (start(r), if last { (r + 1) * sb_px } else { start(r + 1) })
}

/// Loop restoration stripes `k0..=k1` restored in the band `lum0..lum1`
/// (stripe `k` starts at luma row `64 * k - 8`).
fn band_stripes(lum0: usize, lum1: usize, frame_h: usize) -> (usize, usize) {
    let k0 = (lum0 + 8) / 64;
    let mut k1 = k0;
    while 64 * (k1 + 1) < lum1 + 8 && 64 * (k1 + 1) < frame_h + 8 {
        k1 += 1;
    }
    (k0, k1)
}

/// Mode info rows `m0..m1` a band job keeps for superblock row `r`:
/// deblocking reads the row above the superblock row (two luma rows for
/// subsampled chroma), the band starts 2 rows above it.
fn band_mi_rows(f: &FrameCtx, r: usize) -> (usize, usize) {
    (
        (r * f.sb4).saturating_sub(2),
        ((r + 1) * f.sb4).min(f.mi_rows),
    )
}

/// Sets up `dst` for the mode info rows of superblock row `r`
/// ([`band_mi_rows`], whole frame width) with the same per-block arrays
/// as `like`.
fn reset_band_mi(dst: &mut MiWin, like: &MiWin, f: &FrameCtx, r: usize) {
    let (m0, m1) = band_mi_rows(f, r);
    dst.reset(
        f.sb_cols * f.sb4 + 32,
        (m1 - m0).next_power_of_two(),
        0,
        m0,
        f.ss_x,
        f.ss_y,
        !like.lf_tx[0].is_empty(),
        !like.lf_tx[1].is_empty(),
        !like.delta_lf.is_empty(),
        false,
    );
}

/// Copies mode info rows `m0..m1`, columns `c0..c1` (luma 4x4 units, even)
/// with their deblocking transform sizes and delta LF values from `src`
/// to `dst` (which may start at other columns).
#[allow(clippy::too_many_arguments)]
fn copy_mi_rows(
    dst: &mut MiWin,
    src: &MiWin,
    m0: usize,
    m1: usize,
    c0: usize,
    c1: usize,
    ss_x: usize,
    ss_y: usize,
) {
    let n = c1 - c0;
    let (dc, sc) = (c0 - dst.col0, c0 - src.col0);
    for row in m0..m1 {
        dst.row_mut(row)[dc..dc + n].copy_from_slice(&src.row(row)[sc..sc + n]);
        if !src.lf_tx[0].is_empty() {
            dst.lf_tx_row_mut(0, row)[dc..dc + n]
                .copy_from_slice(&src.lf_tx_row(0, row)[sc..sc + n]);
        }
        if !src.delta_lf.is_empty() {
            let (d, s) = (dst.idx(row, c0), src.idx(row, c0));
            dst.delta_lf[d..d + n].copy_from_slice(&src.delta_lf[s..s + n]);
        }
    }
    if !src.lf_tx[1].is_empty() {
        let (dc, sc, n) = (dc >> ss_x, sc >> ss_x, (c1 >> ss_x) - (c0 >> ss_x));
        for r4 in (m0 >> ss_y)..((m1 + ss_y) >> ss_y) {
            dst.lf_tx_row_mut(1, r4)[dc..dc + n].copy_from_slice(&src.lf_tx_row(1, r4)[sc..sc + n]);
        }
    }
}

/// Copies the `cdef_idx` rows of the band of superblock row `r` (from the
/// frame's `cdef_idx`, `bp.cdef_stride` wide) and the restoration unit rows
/// of its stripes (from the frame's units `lr`) into `bb`.
fn fill_band_meta<T: Pixel>(
    f: &FrameCtx,
    bp: &BandParams,
    bb: &mut BandBufs<T>,
    r: usize,
    last: bool,
    cdef_idx: &[i8],
    lr: &[LrPlane; 3],
) {
    let (lum0, lum1) = band_rows(r, last, f.sb_px);
    let (k0, k1) = band_stripes(lum0, lum1, bp.frame_h);
    for p in (0..f.num_planes).filter(|&p| bp.lr_on[p]) {
        let sy = f.sub(p).1;
        let lp = &lr[p];
        let dl = &mut bb.lr[p];
        (dl.unit_size, dl.unit_rows, dl.unit_cols) = (lp.unit_size, lp.unit_rows, lp.unit_cols);
        dl.units.resize(lp.units.len(), Default::default());
        let mut prev_ur = usize::MAX;
        for k in k0..=k1 {
            // The unit row `lr_stripe` uses for stripe k.
            let ur = (lp.unit_rows - 1).min(((64 * k) >> sy) / lp.unit_size);
            if ur != prev_ur {
                prev_ur = ur;
                let row = ur * lp.unit_cols..(ur + 1) * lp.unit_cols;
                dl.units[row.clone()].copy_from_slice(&lp.units[row]);
            }
        }
    }
    if bp.cdef {
        let mi_row0 = lum0 / 4;
        let mi_row1 = (lum1 / 4).min(f.mi_rows);
        let (cr0, cr1) = (mi_row0 >> 4, ((mi_row1 - 1) >> 4) + 1);
        let cs = bp.cdef_stride;
        bb.cdef_idx.clear();
        bb.cdef_idx.extend_from_slice(&cdef_idx[cr0 * cs..cr1 * cs]);
        bb.cdef_r0 = cr0;
    }
}

/// A decoded superblock row of [`Pipeline::run_bands`] waiting for the
/// finishing stage.
struct RowJob<T: Pixel> {
    r: usize,
    last: bool,
    bufs: Box<BandBufs<T>>,
}

/// State of the sequential finishing stage of [`Pipeline::run_bands`]
/// (see [`BandWorker::finish_row`]).
struct FinishState<T: Pixel> {
    /// The last `BAND_TOP` deblocked rows of the previous row.
    carry: [PlaneWin<T>; 3],
    /// Saved stripe boundaries still needed by later bands.
    bounds: VecDeque<std::sync::Arc<LrBoundary<T>>>,
    /// Recycled stripe boundaries.
    free_bounds: Vec<LrBoundary<T>>,
    /// Next stripe boundary to save.
    next_bound: usize,
    /// Deblocking edge parameter scratch.
    lf_grid: Vec<u8>,
}

enum BandTask<T: Pixel> {
    Finish(RowJob<T>),
    Band(BandJob<T>),
    /// Decoding unit (see [`UnitRunner`]) of lane (index, unit).
    Unit(usize, usize),
}

/// Decoding units `next, next + step, ..` (below `end`) that run one at a
/// time in this order (the superblock rows of one tile).
struct UnitLane {
    next: usize,
    end: usize,
    step: usize,
    /// A unit of the lane is running.
    busy: bool,
}

enum SpinPoll<T: Pixel> {
    Task(BandTask<T>),
    Idle,
    Done,
    Contended,
}

/// Producer of the rows of a [`BandPipeline`] that runs on the workers:
/// decoding units (handed out in the order of their lanes, see
/// [`BandPipeline::set_units`]) that queue the decoded rows in order with
/// [`BandPipeline::push_row`].
trait UnitRunner<T: Pixel>: Sync {
    /// Runs unit `u` on `lane`; returns `(ok, lane_released)`.
    fn run_unit(
        &self,
        w: &mut BandWorker<'_, T>,
        bp: &BandParams,
        lane: usize,
        u: usize,
        pipe: &BandPipeline<T>,
    ) -> (bool, bool);
}

struct BandQueue<T: Pixel> {
    /// Decoded rows, in order.
    rows: VecDeque<RowJob<T>>,
    /// A worker runs the finishing stage.
    finishing: bool,
    bands: VecDeque<BandJob<T>>,
    bands_running: usize,
    /// Recycled buffers.
    free: Vec<Box<BandBufs<T>>>,
    /// Recycled stripe boundaries returned by completed bands.
    free_bounds: Vec<LrBoundary<T>>,
    /// No more rows will be queued.
    closed: bool,
    /// Decoding units: lanes of units that run in order (see
    /// [`BandPipeline::set_units`]), units running.
    lanes: Vec<UnitLane>,
    units_running: usize,
    /// The earliest unit that failed (`usize::MAX`: none): later units are
    /// not handed out any more, earlier ones still run (so that the error
    /// of the first failing unit in decoding order is found).
    stop_at: usize,
    /// Bands queued beyond this many are processed before new units.
    band_backlog: usize,
    /// Whether bands have filter work (`cdef` or `lr`) that can run in
    /// parallel across workers before acquiring `out_planes`.
    parallel_bands: bool,
    /// A task panicked: the workers stop (see [`AbortOnPanic`]).
    aborted: bool,
}

/// Unblocks the other workers of a [`BandPipeline`] if a task (or the
/// thread feeding the pipeline, see [`Pipeline::run_bands`]) panics,
/// instead of leaving them waiting forever for its result (the panic then
/// propagates out of the thread scope).
struct AbortOnPanic<'a, T: Pixel>(&'a BandPipeline<T>);

impl<T: Pixel> Drop for AbortOnPanic<'_, T> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0
                .q
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .aborted = true;
            self.0.cv.notify_all();
        }
    }
}

struct OutState<T: Pixel> {
    next_lum0: usize,
    writing: bool,
    pending: Vec<BandJob<T>>,
}

/// Work queue of [`Pipeline::run_bands`]: decoded rows go through the
/// finishing stage one at a time in order, the resulting bands are
/// processed by any worker.
struct BandPipeline<T: Pixel> {
    q: Mutex<BandQueue<T>>,
    cv: std::sync::Condvar,
    fin: Mutex<FinishState<T>>,
    out_planes: Mutex<[PlaneWin<T>; 3]>,
    out_state: Mutex<OutState<T>>,
    returned_scratches: Mutex<Vec<BandWorkerScratch<T>>>,
}

impl<T: Pixel> BandPipeline<T> {
    fn with_pooled(
        out_planes: [PlaneWin<T>; 3],
        free: Vec<Box<BandBufs<T>>>,
        carry: [PlaneWin<T>; 3],
        lf_grid: Vec<u8>,
        free_bounds: Vec<LrBoundary<T>>,
        band_backlog: usize,
    ) -> Self {
        Self {
            q: Mutex::new(BandQueue {
                rows: VecDeque::new(),
                finishing: false,
                bands: VecDeque::new(),
                bands_running: 0,
                free,
                free_bounds: Vec::new(),
                closed: false,
                lanes: Vec::new(),
                units_running: 0,
                stop_at: usize::MAX,
                band_backlog,
                parallel_bands: true,
                aborted: false,
            }),
            cv: std::sync::Condvar::new(),
            fin: Mutex::new(FinishState {
                carry,
                bounds: VecDeque::new(),
                free_bounds,
                next_bound: 1,
                lf_grid,
            }),
            out_planes: Mutex::new(out_planes),
            out_state: Mutex::new(OutState {
                next_lum0: 0,
                writing: false,
                pending: Vec::new(),
            }),
            returned_scratches: Mutex::new(Vec::with_capacity(8)),
        }
    }

    fn set_parallel_bands(&self, parallel_bands: bool) {
        self.q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .parallel_bands = parallel_bands;
    }

    fn return_scratch(&self, sc: BandWorkerScratch<T>) {
        let mut guard = self
            .returned_scratches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.len() < 16 {
            guard.push(sc);
        }
    }

    fn reclaim_into(
        &self,
        free_out: &mut Vec<Box<BandBufs<T>>>,
        carry_out: &mut [PlaneWin<T>; 3],
        lf_grid_out: &mut Vec<u8>,
        lr_free_out: &mut Vec<LrBoundary<T>>,
        scratches_out: &mut Vec<BandWorkerScratch<T>>,
    ) -> [PlaneWin<T>; 3] {
        let mut q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let drained_rows: Vec<_> = q.rows.drain(..).collect();
        for r in drained_rows {
            q.free.push(r.bufs);
        }
        let drained_bands: Vec<_> = q.bands.drain(..).collect();
        let drained_out: Vec<_> = self
            .out_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .drain(..)
            .collect();
        for b in drained_bands.into_iter().chain(drained_out) {
            for arc in b.bounds {
                if let Ok(bnd) = std::sync::Arc::try_unwrap(arc) {
                    q.free_bounds.push(bnd);
                }
            }
            match b.bufs {
                BandBuf::Owned(bufs) => q.free.push(bufs),
                BandBuf::Shared(arc) => {
                    if let Ok(bufs) = std::sync::Arc::try_unwrap(arc) {
                        q.free.push(bufs);
                    }
                }
            }
        }
        *free_out = std::mem::take(&mut q.free);
        let mut fin = self
            .fin
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let drained_bounds: Vec<_> = fin.bounds.drain(..).collect();
        for arc in drained_bounds {
            if let Ok(bnd) = std::sync::Arc::try_unwrap(arc) {
                fin.free_bounds.push(bnd);
            }
        }
        fin.free_bounds.append(&mut q.free_bounds);
        *lr_free_out = std::mem::take(&mut fin.free_bounds);
        *carry_out = std::mem::take(&mut fin.carry);
        *lf_grid_out = std::mem::take(&mut fin.lf_grid);
        drop(fin);
        drop(q);
        {
            let mut scs = self
                .returned_scratches
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for sc in scs.drain(..) {
                if scratches_out.len() < 16 {
                    scratches_out.push(sc);
                }
            }
        }
        let mut out_planes = self
            .out_planes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *out_planes)
    }

    /// Hands out decoding units to the workers (instead of rows being
    /// pushed from outside): every lane `(start, end)` runs units `start,
    /// start + step, ..` below `end` one at a time in this order, lanes run
    /// in parallel (earliest unit first); the pipeline closes once all of
    /// them ran. Bands queued beyond `band_backlog` go first.
    fn set_units(&self, lanes: Vec<(usize, usize)>, step: usize, band_backlog: usize) {
        let mut q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.lanes = lanes
            .into_iter()
            .filter(|&(start, end)| start < end)
            .map(|(next, end)| UnitLane {
                next,
                end,
                step,
                busy: false,
            })
            .collect();
        q.band_backlog = band_backlog;
        q.closed = q.lanes.is_empty();
    }

    /// Buffers for the next row (recycled if possible), without blocking on
    /// the queue lock so the main thread is not put to sleep.
    fn take_bufs(&self) -> Box<BandBufs<T>> {
        loop {
            match self.q.try_lock() {
                Ok(mut q) => {
                    let mut b: Box<BandBufs<T>> = q.free.pop().unwrap_or_default();
                    b.lf_v_done = false;
                    return b;
                }
                Err(std::sync::TryLockError::Poisoned(p)) => {
                    let mut b: Box<BandBufs<T>> = p.into_inner().free.pop().unwrap_or_default();
                    b.lf_v_done = false;
                    return b;
                }
                Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
            }
        }
    }

    fn abort(&self) {
        self.q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .aborted = true;
        self.cv.notify_all();
    }

    fn is_aborted(&self) -> bool {
        self.q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .aborted
    }

    /// Queues a decoded row; returns the number of queued/running rows and bands.
    fn push_row(&self, job: RowJob<T>) -> usize {
        let mut q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.rows.push_back(job);
        let n = q.rows.len() + q.bands.len() + q.bands_running;
        drop(q);
        self.cv.notify_one();
        n
    }

    /// Queues a band directly for CDEF / output processing.
    fn push_band(&self, job: BandJob<T>) {
        let mut q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pos = q
            .bands
            .iter()
            .position(|b| b.lum0 > job.lum0)
            .unwrap_or(q.bands.len());
        q.bands.insert(pos, job);
        drop(q);
        self.cv.notify_one();
    }

    fn recycle_band(&self, job: BandJob<T>) {
        let BandJob { bufs, bounds, .. } = job;
        let mut q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for arc in bounds {
            if let Ok(bnd) = std::sync::Arc::try_unwrap(arc) {
                q.free_bounds.push(bnd);
            }
        }
        match bufs {
            BandBuf::Owned(bufs) => q.free.push(bufs),
            BandBuf::Shared(arc) => {
                if let Ok(bufs) = std::sync::Arc::try_unwrap(arc) {
                    q.free.push(bufs);
                }
            }
        }
    }

    fn append_band_planes(&self, f: &FrameCtx, bp: &BandParams, job: &BandJob<T>) {
        let np = f.num_planes;
        let (lum0, lum1) = (job.lum0, job.lum1);
        let b: &BandBufs<T> = match &job.bufs {
            BandBuf::Owned(b) => b,
            BandBuf::Shared(b) => b,
        };
        let mut out_guard = self
            .out_planes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for p in 0..np {
            let sy = f.sub(p).1;
            let stride = out_guard[p].stride;
            let w = bp.out_w[p];
            let oy0 = lum0 >> sy;
            let y1 = (lum1 >> sy).min(bp.mi_h[p]).min(bp.out_h[p]);
            let out_buf = out_guard[p].buf_mut();
            let total = bp.out_h[p] * stride;
            if out_buf.capacity() < total {
                out_buf.reserve_exact(total - out_buf.len());
            }
            if y1 > oy0 {
                let start = oy0 * stride;
                if stride == w && out_buf.len() == start {
                    for y in oy0..y1 {
                        out_buf.extend_from_slice(&b.planes[p].row(y)[..w]);
                    }
                } else {
                    let end = (y1 - 1) * stride + w;
                    if out_buf.len() < end {
                        out_buf.resize(end, T::default());
                    }
                    for y in oy0..y1 {
                        let d = y * stride;
                        out_buf[d..d + w].copy_from_slice(&b.planes[p].row(y)[..w]);
                    }
                }
            }
        }
    }

    /// Writes a non-LR band into `out_planes` strictly in `lum0` order so
    /// `out_buf.extend_from_slice` never triggers an out-of-order `resize`
    /// zero-fill under the lock.
    fn write_band(&self, f: &FrameCtx, bp: &BandParams, job: BandJob<T>) {
        let mut st = self
            .out_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if job.lum0 != st.next_lum0 || st.writing {
            st.pending.push(job);
            return;
        }
        st.writing = true;
        drop(st);
        let mut cur = job;
        loop {
            let next_lum = cur.lum1;
            self.append_band_planes(f, bp, &cur);
            self.recycle_band(cur);
            let mut st = self
                .out_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            st.next_lum0 = next_lum;
            if let Some(pos) = st.pending.iter().position(|b| b.lum0 == next_lum) {
                cur = st.pending.swap_remove(pos);
                drop(st);
            } else {
                st.writing = false;
                break;
            }
        }
    }

    fn queued_tasks(&self) -> usize {
        let q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.rows.len() + q.bands.len() + q.bands_running
    }

    /// Runs the available tasks without ever blocking (not even on the
    /// queue lock), spinning instead of sleeping until all pipeline work
    /// and worker threads have finished. Used on the thread that decodes:
    /// a sleeping thread may be woken on a slower core (see
    /// `Pipeline::run_bands_pipe`).
    fn work_spin(
        &self,
        w: &mut BandWorker<'_, T>,
        bp: &BandParams,
        units: Option<&dyn UnitRunner<T>>,
        threads_done: impl Fn() -> bool,
    ) {
        let _abort = AbortOnPanic(self);
        loop {
            match self.try_next() {
                SpinPoll::Task(t) => {
                    self.run(w, bp, t, units);
                    continue;
                }
                SpinPoll::Done => break,
                SpinPoll::Idle | SpinPoll::Contended => {}
            }
            // Back off so that the workers get the queue lock.
            for _ in 0..64 {
                std::hint::spin_loop();
            }
        }
        while !threads_done() {
            for _ in 0..64 {
                std::hint::spin_loop();
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
    }

    /// No more rows will be queued.
    fn close(&self) {
        self.q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .closed = true;
        self.cv.notify_all();
    }

    /// The next task; with `wait`, blocks until there is one or all work
    /// is done.
    fn next(&self, wait: bool) -> Option<BandTask<T>> {
        let mut q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if q.aborted {
                return None;
            }
            if let Some(t) = Self::pick(&mut q) {
                return Some(t);
            }
            if !wait || (q.closed && q.rows.is_empty() && !q.finishing && q.bands.is_empty()) {
                return None;
            }
            q = self
                .cv
                .wait(q)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Non-blocking poll for `work_spin`: checks task availability and full
    /// pipeline completion atomically under `self.q` so a row pushed right
    /// before `close()` cannot be missed.
    fn try_next(&self) -> SpinPoll<T> {
        let mut q = match self.q.try_lock() {
            Ok(q) => q,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return SpinPoll::Contended,
        };
        if q.aborted {
            return SpinPoll::Done;
        }
        if let Some(t) = Self::pick(&mut q) {
            return SpinPoll::Task(t);
        }
        if q.closed
            && q.rows.is_empty()
            && !q.finishing
            && q.bands.is_empty()
            && q.bands_running == 0
            && q.units_running == 0
        {
            return SpinPoll::Done;
        }
        SpinPoll::Idle
    }

    /// Claims the next available task, if any.
    fn pick(q: &mut BandQueue<T>) -> Option<BandTask<T>> {
        {
            if !q.finishing {
                if let Some(row) = q.rows.pop_front() {
                    q.finishing = true;
                    return Some(BandTask::Finish(row));
                }
            }
            // The ready lane with the earliest unit.
            let stop_at = q.stop_at;
            let lane = q
                .lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| !l.busy && l.next < l.end.min(stop_at))
                .min_by_key(|(_, l)| l.next)
                .map(|(i, _)| i);
            let can_band = q.parallel_bands || q.bands_running == 0;
            if can_band && (lane.is_none() || q.bands.len() > q.band_backlog) {
                if let Some(b) = q.bands.pop_front() {
                    q.bands_running += 1;
                    return Some(BandTask::Band(b));
                }
            }
            if let Some(i) = lane {
                let l = &mut q.lanes[i];
                let u = l.next;
                l.next += l.step;
                l.busy = true;
                q.units_running += 1;
                return Some(BandTask::Unit(i, u));
            }
            None
        }
    }

    fn release_lane(&self, lane: usize) {
        let mut q = self
            .q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.lanes[lane].busy = false;
        let stop_at = q.stop_at;
        let has_next = q.lanes[lane].next < q.lanes[lane].end.min(stop_at);
        drop(q);
        if has_next {
            self.cv.notify_one();
        }
    }

    fn run(
        &self,
        w: &mut BandWorker<'_, T>,
        bp: &BandParams,
        task: BandTask<T>,
        units: Option<&dyn UnitRunner<T>>,
    ) {
        let _abort = AbortOnPanic(self);
        match task {
            BandTask::Unit(lane, u) => {
                let (ok, released) =
                    units.map_or((false, false), |units| units.run_unit(w, bp, lane, u, self));
                let mut q = self
                    .q
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                q.units_running -= 1;
                if !released {
                    q.lanes[lane].busy = false;
                }
                if !ok {
                    q.stop_at = q.stop_at.min(u);
                }
                let stop_at = q.stop_at;
                if q.units_running == 0 && q.lanes.iter().all(|l| l.next >= l.end.min(stop_at)) {
                    q.closed = true;
                }
                drop(q);
                self.cv.notify_all();
            }
            BandTask::Finish(row) => {
                let mut fin = self
                    .fin
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut new_bands = Vec::with_capacity(3);
                w.finish_row(bp, &mut fin, row, &mut new_bands);
                let mut q = self
                    .q
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if !q.free_bounds.is_empty() {
                    fin.free_bounds.append(&mut q.free_bounds);
                }
                drop(fin);
                q.bands.extend(new_bands);
                q.finishing = false;
                drop(q);
                self.cv.notify_all();
            }
            BandTask::Band(b) => {
                let recycled = w.process(bp, b, self);
                let mut q = self
                    .q
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                q.bands_running -= 1;
                if let Some(BandJob { bufs, bounds, .. }) = recycled {
                    for arc in bounds {
                        if let Ok(bnd) = std::sync::Arc::try_unwrap(arc) {
                            q.free_bounds.push(bnd);
                        }
                    }
                    match bufs {
                        BandBuf::Owned(bufs) => q.free.push(bufs),
                        BandBuf::Shared(arc) => {
                            if let Ok(bufs) = std::sync::Arc::try_unwrap(arc) {
                                q.free.push(bufs);
                            }
                        }
                    }
                }
                let notify = !q.parallel_bands && q.bands_running == 0 && !q.bands.is_empty();
                let done = q.closed
                    && q.rows.is_empty()
                    && !q.finishing
                    && q.bands.is_empty()
                    && q.bands_running == 0
                    && q.units_running == 0;
                drop(q);
                if done {
                    self.cv.notify_all();
                } else if notify {
                    self.cv.notify_one();
                }
            }
        }
    }

    /// Runs tasks until all work is done (`help`: finishes any queued rows
    /// and runs bands while backlog exceeds `band_backlog`, without waiting).
    /// `units` runs the decoding units, if any.
    fn work(
        &self,
        w: &mut BandWorker<'_, T>,
        bp: &BandParams,
        help: bool,
        units: Option<&dyn UnitRunner<T>>,
    ) {
        // Also covers panics outside of `run` (in `next`).
        let _abort = AbortOnPanic(self);
        if help {
            loop {
                let t = {
                    let mut q = self
                        .q
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if q.aborted {
                        None
                    } else if !q.finishing && !q.rows.is_empty() {
                        q.finishing = true;
                        q.rows.pop_front().map(BandTask::Finish)
                    } else if (q.parallel_bands || q.bands_running == 0)
                        && q.bands.len() > q.band_backlog
                    {
                        q.bands_running += 1;
                        q.bands.pop_front().map(BandTask::Band)
                    } else {
                        None
                    }
                };
                let Some(t) = t else { break };
                self.run(w, bp, t, units);
            }
            return;
        }
        while let Some(t) = self.next(true) {
            self.run(w, bp, t, units);
        }
    }
}

/// Owned buffers of one post-filter band (see [`Pipeline::run_bands`]),
/// recycled between bands.
#[derive(Default)]
pub(crate) struct BandBufs<T: Pixel> {
    /// The row buffer superblock row `r` was decoded and deblocked in:
    /// luma rows `r * sb_px - BAND_TOP .. (r + 1) * sb_px` (from 0 for the
    /// first row), which contain the band's rows and their CDEF borders.
    planes: [PlaneWin<T>; 3],
    /// The 2 rows above the band (pre-CDEF).
    above: [[Vec<T>; 2]; 3],
    /// Mode info rows of the band.
    mi: MiWin,
    /// `cdef_idx` rows of the band (`cdef_stride` wide, from 64x64 row
    /// `cdef_r0`).
    cdef_idx: Vec<i8>,
    cdef_r0: usize,
    /// Loop restoration units (only the unit rows of the band's stripes
    /// are filled in).
    lr: [LrPlane; 3],
    /// Whether all edges of the row but the horizontal ones on its top
    /// boundary are already deblocked (see `TileJobs::run_unit`), leaving
    /// only those to `BandWorker::finish_row`.
    lf_v_done: bool,
    /// Pooled neighbor-row carry buffers for `RowOrder`.
    carry_planes: [PlaneWin<T>; 3],
    carry_mi: MiWin,
}

enum BandBuf<T: Pixel> {
    Owned(Box<BandBufs<T>>),
    Shared(std::sync::Arc<Box<BandBufs<T>>>),
}

/// One band of output rows `lum0..lum1` (luma) post-filtered by a worker:
/// CDEF in `bufs.planes`, then loop restoration (or a copy) into the
/// band's rows of `BandPipeline::out_planes`.
struct BandJob<T: Pixel> {
    lum0: usize,
    lum1: usize,
    bufs: BandBuf<T>,
    /// Loop restoration boundaries `k0..=k1 + 1` of the band's stripes
    /// `k0..=k1` that exist.
    bounds: Vec<std::sync::Arc<LrBoundary<T>>>,
}

/// Frame constants needed by the band workers.
#[derive(Clone, Copy)]
struct BandParams {
    deblock: bool,
    cdef: bool,
    lr: bool,
    lr_on: [bool; 3],
    mi_w: [usize; 3],
    mi_h: [usize; 3],
    out_w: [usize; 3],
    out_h: [usize; 3],
    out_stride: [usize; 3],
    cdef_stride: usize,
    frame_h: usize,
    col_range: (usize, usize),
}

struct BandWorker<'a, T: Pixel> {
    f: FrameCtx<'a>,
    dsp: crate::decoder::av1::dsp::Dsp<T>,
    sc: BandWorkerScratch<T>,
}

impl<'a, T: Pixel> BandWorker<'a, T> {
    fn new(fd: &'a FrameData) -> Self {
        Self::with_scratch(&fd.seq, &fd.frame, BandWorkerScratch::default())
    }

    fn with_scratch(
        seq: &'a crate::decoder::av1::headers::SequenceHeader,
        fh: &'a crate::decoder::av1::headers::FrameHeader,
        sc: BandWorkerScratch<T>,
    ) -> Self {
        let f = FrameCtx::new(seq, fh);
        let dsp = crate::decoder::av1::dsp::Dsp::new(f.bd);
        Self { f, dsp, sc }
    }

    fn into_scratch(self) -> BandWorkerScratch<T> {
        self.sc
    }

    /// The finishing stage of row `job.r` (rows in order): fills the rows
    /// above the superblock row into its buffer from the previous row,
    /// deblocks the row, keeps its last rows for the next one and saves the
    /// stripe boundaries that became final. Pushes the row's band(s).
    fn finish_row(
        &mut self,
        bp: &BandParams,
        fs: &mut FinishState<T>,
        job: RowJob<T>,
        out_bands: &mut Vec<BandJob<T>>,
    ) {
        let f = &self.f;
        let np = f.num_planes;
        let (sb_px, sb4) = (f.sb_px, f.sb4);
        let RowJob { r, last, mut bufs } = job;
        let (lum0, lum1) = band_rows(r, last, sb_px);
        let (k0, k1) = band_stripes(lum0, lum1, bp.frame_h);
        let b = &mut *bufs;
        if r > 0 {
            for p in 0..np {
                let (c, w) = (&fs.carry[p], bp.mi_w[p]);
                for y in c.y0..c.y0 + c.rows {
                    b.planes[p].row_mut(y)[..w].copy_from_slice(&c.row(y)[..w]);
                }
            }
        }
        if bp.deblock {
            let passes = if b.lf_v_done {
                deblock::PASS_H_TOP
            } else {
                deblock::PASS_V | deblock::PASS_H
            };
            b.lf_v_done = false;
            deblock::deblock_rows_passes(
                f,
                &b.mi,
                &mut b.planes[..np],
                r * sb4,
                (r + 1) * sb4,
                bp.col_range,
                &self.dsp,
                &mut fs.lf_grid,
                passes,
            );
        }
        for p in 0..np {
            let sy = f.sub(p).1;
            let w = bp.mi_w[p];
            let pl = &b.planes[p];
            if !last {
                let y0 = ((r + 1) * sb_px - BAND_TOP) >> sy;
                let n = BAND_TOP >> sy;
                fs.carry[p].reset(pl.stride, n, 0, y0);
                for y in y0..y0 + n {
                    fs.carry[p].row_mut(y)[..w].copy_from_slice(&pl.row(y)[..w]);
                }
            }
            let y0 = lum0 >> sy;
            if bp.cdef && y0 >= 2 {
                for (i, a) in b.above[p].iter_mut().enumerate() {
                    a.clear();
                    a.extend_from_slice(&pl.row(y0 - 2 + i)[..w]);
                }
            }
        }
        if bp.lr {
            let lim = (f.mi_rows * 4).min((r + 1) * sb_px);
            loop {
                let k = fs.next_bound;
                if 64 * k - 8 >= lim || (!last && 64 * k - 10 >= lum1) {
                    break;
                }
                let mut bnd = fs.free_bounds.pop().unwrap_or_else(|| LrBoundary {
                    k: 0,
                    rows: Default::default(),
                });
                bnd.k = k;
                for p in (0..np).filter(|&p| bp.lr_on[p]) {
                    let sy = f.sub(p).1;
                    let y = (64 * k - 8) >> sy;
                    let w = bp.out_w[p];
                    for (i, dst) in bnd.rows[p].iter_mut().enumerate() {
                        let y = (y + i - 2).min(bp.mi_h[p] - 1);
                        dst.clear();
                        dst.extend_from_slice(&b.planes[p].row(y)[..w]);
                    }
                }
                fs.bounds.push_back(std::sync::Arc::new(bnd));
                fs.next_bound += 1;
            }
        }
        if !bp.cdef && k1 > k0 {
            let shared = std::sync::Arc::new(bufs);
            for k in k0..=k1 {
                let sub_lum0 = if k == k0 { lum0 } else { 64 * k - 8 };
                let sub_lum1 = if k == k1 { lum1 } else { 64 * (k + 1) - 8 };
                let bounds = fs
                    .bounds
                    .iter()
                    .filter(|b| b.k >= k && b.k <= k + 1)
                    .cloned()
                    .collect();
                out_bands.push(BandJob {
                    lum0: sub_lum0,
                    lum1: sub_lum1,
                    bufs: BandBuf::Shared(shared.clone()),
                    bounds,
                });
            }
        } else {
            let bounds = fs
                .bounds
                .iter()
                .filter(|b| b.k >= k0 && b.k <= k1 + 1)
                .cloned()
                .collect();
            out_bands.push(BandJob {
                lum0,
                lum1,
                bufs: BandBuf::Owned(bufs),
                bounds,
            });
        }
        while fs.bounds.front().is_some_and(|b| b.k <= k1) {
            if let Some(arc) = fs.bounds.pop_front() {
                if let Ok(bnd) = std::sync::Arc::try_unwrap(arc) {
                    fs.free_bounds.push(bnd);
                }
            }
        }
    }

    fn process(
        &mut self,
        bp: &BandParams,
        mut job: BandJob<T>,
        pipe: &BandPipeline<T>,
    ) -> Option<BandJob<T>> {
        let f = &self.f;
        let np = f.num_planes;
        let (lum0, lum1) = (job.lum0, job.lum1);
        let mut ranges = [(0usize, 0usize); 3];
        for p in 0..np {
            let sy = f.sub(p).1;
            ranges[p] = (lum0 >> sy, (lum1 >> sy).min(bp.mi_h[p]));
        }
        // Shared buffers are only built without CDEF.
        debug_assert!(!bp.cdef || matches!(job.bufs, BandBuf::Owned(_)));
        if let BandBuf::Owned(b) = &mut job.bufs {
            let b = &mut **b;
            if b.lf_v_done {
                b.lf_v_done = false;
                if bp.deblock {
                    let r = (lum0 + POST_LAG) / f.sb_px;
                    deblock::deblock_rows_passes(
                        f,
                        &b.mi,
                        &mut b.planes[..np],
                        r * f.sb4,
                        (r + 1) * f.sb4,
                        bp.col_range,
                        &self.dsp,
                        &mut self.sc.lf_grid,
                        deblock::PASS_H_TOP,
                    );
                }
                if bp.cdef {
                    for p in 0..np {
                        let sy = f.sub(p).1;
                        let w = bp.mi_w[p];
                        let y0 = lum0 >> sy;
                        if y0 >= 2 {
                            for (i, a) in b.above[p].iter_mut().enumerate() {
                                a.clear();
                                a.extend_from_slice(&b.planes[p].row(y0 - 2 + i)[..w]);
                            }
                        }
                    }
                }
            }
        }
        if let (true, BandBuf::Owned(b)) = (bp.cdef, &mut job.bufs) {
            let b = &mut **b;
            let mut dims = [(0usize, 0usize); 3];
            for p in 0..np {
                dims[p] = (bp.mi_w[p], bp.mi_h[p]);
            }
            let (idx, cs, r0) = (&b.cdef_idx, bp.cdef_stride, b.cdef_r0);
            cdef_apply::cdef_sbrow(
                f,
                &b.mi,
                |r, c| idx[((r >> 4) - r0) * cs + (c >> 4)],
                &mut b.planes[..np],
                &mut self.sc.cdef_srcs,
                &b.above[..np],
                &ranges[..np],
                &dims[..np],
                lum0 / 4,
                lum1 / 4,
                (0, f.sb_cols * f.sb4),
                &self.dsp,
            );
        }
        if !bp.lr {
            pipe.write_band(&self.f, bp, job);
            return None;
        }
        let b: &BandBufs<T> = match &job.bufs {
            BandBuf::Owned(b) => b,
            BandBuf::Shared(b) => b,
        };
        let mut lr_rows = [0usize; 3];
        let lim = (lum1 + 8).min(bp.frame_h + 8);
        for p in 0..np {
            if !bp.lr_on[p] {
                continue;
            }
            let sy = f.sub(p).1;
            let w = bp.out_w[p];
            let out_h = bp.out_h[p];
            let oy0 = lum0 >> sy;
            let mut k = (lum0 + 8) / 64;
            if 64 * k >= lim || w == 0 || oy0 >= out_h {
                continue;
            }
            let k_last = (lim - 1) / 64;
            let end_y_last = (((64 * k_last as isize - 8) >> sy) + (64 >> sy)).max(0) as usize;
            let cap_rows = end_y_last.min(out_h).saturating_sub(oy0);
            if cap_rows == 0 {
                continue;
            }
            let need = cap_rows * w;
            if self.sc.lr_band_out[p].len() < need {
                self.sc.lr_band_out[p].resize(need, T::default());
            }
            let mut max_y = oy0;
            while 64 * k < lim {
                let start_y = (64 * k as isize - 8) >> sy;
                let end_y = start_y + (64 >> sy) - 1;
                let g = StripeGeom {
                    start_y,
                    end_y,
                    plane_w: w,
                    plane_h: out_h,
                };
                let y_first = start_y.max(0) as usize;
                let y_end = ((end_y + 1).max(0) as usize).min(out_h);
                if y_end > y_first {
                    max_y = max_y.max(y_end);
                    let above = job.bounds.iter().find(|b| b.k == k).map(|b| &b.rows[p]);
                    let below = job.bounds.iter().find(|b| b.k == k + 1).map(|b| &b.rows[p]);
                    lr_apply::lr_stripe(
                        &b.lr[p],
                        &g,
                        &b.planes[p],
                        above,
                        below,
                        &mut self.sc.lr_band_out[p][(y_first - oy0) * w..],
                        w,
                        &mut self.sc.scratch,
                        sy,
                        f.bd,
                        (0, usize::MAX),
                        &self.dsp,
                    );
                }
                k += 1;
            }
            lr_rows[p] = max_y - oy0;
        }
        let mut out_guard = pipe
            .out_planes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for p in 0..np {
            let sy = f.sub(p).1;
            let stride = out_guard[p].stride;
            let w = bp.out_w[p];
            let oy0 = lum0 >> sy;
            let out_buf = out_guard[p].buf_mut();
            let total = bp.out_h[p] * stride;
            if out_buf.capacity() < total {
                out_buf.reserve_exact(total - out_buf.len());
            }
            if !bp.lr_on[p] {
                let y1 = ranges[p].1.min(bp.out_h[p]);
                if y1 > oy0 {
                    let start = oy0 * stride;
                    if stride == w && out_buf.len() == start {
                        for y in oy0..y1 {
                            out_buf.extend_from_slice(&b.planes[p].row(y)[..w]);
                        }
                    } else {
                        let end = (y1 - 1) * stride + w;
                        if out_buf.len() < end {
                            out_buf.resize(end, T::default());
                        }
                        for y in oy0..y1 {
                            let d = y * stride;
                            out_buf[d..d + w].copy_from_slice(&b.planes[p].row(y)[..w]);
                        }
                    }
                }
            } else {
                let n_rows = lr_rows[p];
                if n_rows > 0 {
                    let src = &self.sc.lr_band_out[p];
                    let start = oy0 * stride;
                    if stride == w && out_buf.len() == start {
                        out_buf.extend_from_slice(&src[..n_rows * w]);
                    } else {
                        let end = (oy0 + n_rows - 1) * stride + w;
                        if out_buf.len() < end {
                            out_buf.resize(end, T::default());
                        }
                        for r in 0..n_rows {
                            let d = (oy0 + r) * stride;
                            let s = r * w;
                            out_buf[d..d + w].copy_from_slice(&src[s..s + w]);
                        }
                    }
                }
            }
        }
        Some(job)
    }
}

/// Persistent decoding state of one tile in [`Pipeline::run_tiles_bands`]
/// (set up when its first superblock row is decoded).
struct TileCtx<'d, T: Pixel> {
    /// Tile row the state is set up for (`usize::MAX`: none yet).
    tr: usize,
    ts: Option<TileState<'d>>,
    cdf: Option<Box<CdfContext>>,
    s: FrameState<T>,
    sc: Scratch,
}

impl<'d, T: Pixel> TileCtx<'d, T> {
    fn from_worker(w: TileWorker<T>) -> Self {
        Self {
            tr: usize::MAX,
            ts: None,
            cdf: w.cdf,
            s: w.s,
            sc: w.sc,
        }
    }

    fn into_worker(mut self) -> TileWorker<T> {
        if let Some(ts) = self.ts.take() {
            self.cdf = Some(ts.cdf);
        }
        TileWorker {
            s: self.s,
            sc: self.sc,
            cdf: self.cdf,
        }
    }

    /// Decodes superblock row `r` of tile `(tr, tc)` (coded in `data`) into
    /// `self.s`: planes one superblock row high (the unfiltered last row
    /// is kept in `above_line` for the next row), mode info with the
    /// deblocking arrays `mi_arrays` (luma / chroma transform sizes, delta
    /// LF), a tile local `cdef_idx` (if `cdef`) and frame sized
    /// restoration units of which the tile fills its own.
    #[allow(clippy::too_many_arguments)]
    fn decode_row(
        &mut self,
        f: &FrameCtx,
        dsp: &crate::decoder::av1::dsp::Dsp<T>,
        data: &'d [u8],
        mi_arrays: (bool, bool, bool),
        cdef: bool,
        tr: usize,
        tc: usize,
        r: usize,
    ) -> Result<()> {
        let fh = f.fh;
        let ti = &fh.tile_info;
        let np = f.num_planes;
        let (sb4, sb_px) = (f.sb4, f.sb_px);
        let (mi_r0, mi_r1) = (ti.mi_row_starts[tr], ti.mi_row_starts[tr + 1]);
        let (mi_c0, mi_c1) = (ti.mi_col_starts[tc], ti.mi_col_starts[tc + 1]);
        let (sb_r0, sb_r1) = (mi_r0 / sb4, mi_r1.div_ceil(sb4));
        let tile_mi_al = (mi_c1 - mi_c0).div_ceil(sb4) * sb4;
        let s = &mut self.s;
        if self.tr != tr {
            for p in 0..np {
                let (sx, sy) = f.sub(p);
                let stride = ((tile_mi_al * 4) >> sx).next_multiple_of(64);
                s.planes[p].reset(
                    stride,
                    sb_px >> sy,
                    (mi_c0 * 4) >> sx,
                    (sb_r0 * sb_px) >> sy,
                );
                s.above_line[p].resize(stride + 64, T::default());
            }
            let (lf_y, lf_uv, dlf) = mi_arrays;
            s.mi.reset(
                tile_mi_al + 32,
                2 * sb4,
                mi_c0,
                (sb_r0 * sb4).saturating_sub(sb4),
                f.ss_x,
                f.ss_y,
                lf_y,
                lf_uv,
                dlf,
                false,
            );
            s.above
                .reset(mi_c0, tile_mi_al + 32, np, fh.allow_screen_content_tools);
            s.cdef_idx.clear();
            (s.cdef_stride, s.cdef_row0, s.cdef_col0) = (0, 0, 0);
            if cdef {
                (s.cdef_stride, s.cdef_row0, s.cdef_col0) =
                    (tile_mi_al / 16 + 2, sb_r0 * sb4 / 16, mi_c0 / 16);
                s.cdef_idx
                    .resize(((sb_r1 - sb_r0) * sb4 / 16 + 2) * s.cdef_stride, -1);
            }
            for (p, lp) in s.lr.iter_mut().enumerate() {
                if fh.lr.uses_lr && p < np {
                    lp.reset(fh, p, f.ss_x, f.ss_y);
                } else {
                    lp.units.clear();
                }
            }
            s.ipred = [IntraSetup::new(f, 0), IntraSetup::new(f, 1)];
            if let Some(ts) = self.ts.take() {
                self.cdf = Some(ts.cdf);
            }
            self.ts = Some(TileState::with_cdf(
                data,
                fh,
                f.seq,
                (mi_r0, mi_r1),
                (mi_c0, mi_c1),
                self.cdf.take(),
            ));
            self.tr = tr;
        }
        let Some(ts) = self.ts.as_mut() else {
            return Err(Error::Invalid("tile state not set up"));
        };
        let (mi_row, sb_y) = (r * sb4, r * sb_px);
        s.mi.slide_to(mi_row.saturating_sub(sb4));
        for p in 0..np {
            s.planes[p].y0 = sb_y >> f.sub(p).1;
        }
        s.left.clear();
        s.set_sb_y(f, sb_y);
        let mut dec = SbDecoder {
            f,
            t: ts,
            s: &mut *s,
            sc: &mut self.sc,
            dsp,
            b: Default::default(),
            recon: true,
            sb_y,
            read_deltas: false,
            plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
            stage: stage::DecoderStage::Fused,
            filter_type: [false; 2],
        };
        let mut c = mi_c0;
        while c < mi_c1 {
            dec.decode_sb(mi_row, c)?;
            if dec.t.msac.overrun() {
                return Err(Error::Truncated);
            }
            c += sb4;
        }
        if r + 1 < sb_r1 {
            for p in 0..np {
                let (sx, sy) = f.sub(p);
                let y = ((sb_y + sb_px) >> sy) - 1;
                let w = ((mi_c1 * 4) >> sx) - ((mi_c0 * 4) >> sx);
                let row = &s.planes[p].row(y)[..w];
                s.above_line[p][..w].copy_from_slice(row);
            }
        }
        Ok(())
    }
}

/// A superblock row of [`Pipeline::run_tiles_bands`] being assembled from
/// its tiles.
struct RowSlot<T: Pixel> {
    bufs: Option<Box<BandBufs<T>>>,
    /// Tiles still to be copied in.
    remaining: usize,
}

#[derive(Default)]
struct RowCarry<T: Pixel> {
    planes: [PlaneWin<T>; 3],
    mi: MiWin,
}

fn save_row_carry<T: Pixel>(
    f: &FrameCtx,
    bp: &BandParams,
    r: usize,
    bb: &BandBufs<T>,
    c: &mut RowCarry<T>,
) {
    let np = f.num_planes;
    let sb_px = f.sb_px;
    for p in 0..np {
        let sy = f.sub(p).1;
        let w = bp.mi_w[p];
        let pl = &bb.planes[p];
        let y0 = ((r + 1) * sb_px - BAND_TOP) >> sy;
        let n = BAND_TOP >> sy;
        c.planes[p].reset(w, n, 0, y0);
        for y in y0..y0 + n {
            c.planes[p].row_mut(y)[..w].copy_from_slice(&pl.row(y)[..w]);
        }
    }
    if bp.deblock || bp.cdef {
        let c0 = (r + 1) * f.sb4 - 2;
        let mi = &bb.mi;
        c.mi.reset(
            mi.stride,
            2,
            0,
            c0,
            f.ss_x,
            f.ss_y,
            !mi.lf_tx[0].is_empty(),
            !mi.lf_tx[1].is_empty(),
            !mi.delta_lf.is_empty(),
            false,
        );
        copy_mi_rows(&mut c.mi, mi, c0, c0 + 2, 0, f.mi_cols, f.ss_x, f.ss_y);
    }
}

fn apply_row_carry<T: Pixel>(
    f: &FrameCtx,
    bp: &BandParams,
    r: usize,
    c: &RowCarry<T>,
    bb: &mut BandBufs<T>,
) {
    let np = f.num_planes;
    for p in 0..np {
        let (cp, w) = (&c.planes[p], bp.mi_w[p]);
        for y in cp.y0..cp.y0 + cp.rows {
            bb.planes[p].row_mut(y)[..w].copy_from_slice(&cp.row(y)[..w]);
        }
    }
    if bp.deblock || bp.cdef {
        let (m0, _) = band_mi_rows(f, r);
        copy_mi_rows(
            &mut bb.mi,
            &c.mi,
            m0,
            r * f.sb4,
            0,
            f.mi_cols,
            f.ss_x,
            f.ss_y,
        );
    }
}

fn copy_row_carry_direct<T: Pixel>(
    f: &FrameCtx,
    bp: &BandParams,
    r_next: usize,
    src_bb: &BandBufs<T>,
    dst_bb: &mut BandBufs<T>,
) {
    let np = f.num_planes;
    let sb_px = f.sb_px;
    for p in 0..np {
        let sy = f.sub(p).1;
        let w = bp.mi_w[p];
        let y0 = (r_next * sb_px - BAND_TOP) >> sy;
        let n = BAND_TOP >> sy;
        for y in y0..y0 + n {
            dst_bb.planes[p].row_mut(y)[..w].copy_from_slice(&src_bb.planes[p].row(y)[..w]);
        }
    }
    if bp.deblock || bp.cdef {
        let (m0, _) = band_mi_rows(f, r_next);
        copy_mi_rows(
            &mut dst_bb.mi,
            &src_bb.mi,
            m0,
            r_next * f.sb4,
            0,
            f.mi_cols,
            f.ss_x,
            f.ss_y,
        );
    }
}

/// Completed rows of [`Pipeline::run_tiles_bands`] waiting to be queued in
/// order (with LR) or waiting only on row `r - 1` (without LR).
struct RowOrder<T: Pixel> {
    next: usize,
    ready: Vec<Option<Box<BandBufs<T>>>>,
    /// The last 2 mode info rows of the last queued row (the rows above
    /// the next one).
    carry: MiWin,
    /// Whether row `r - 1` has saved its bottom carry into `carries[r - 1]`
    /// (`above_ready[0]` is always `true`).
    above_ready: Vec<bool>,
    /// Per-row carry from row `r` to row `r + 1` (used when `!bp.lr`).
    carries: Vec<Option<RowCarry<T>>>,
    free_carries: Vec<RowCarry<T>>,
}

/// Frame level `cdef_idx` and restoration units assembled from the tiles.
struct TileMeta {
    cdef_idx: Vec<i8>,
    lr: [LrPlane; 3],
}

/// The decoding units of [`Pipeline::run_tiles_bands`]: unit `u` decodes
/// superblock row `u / tile_cols` of tile column `u % tile_cols`.
struct TileJobs<'d, T: Pixel> {
    fd: &'d FrameData<'d>,
    ctxs: Vec<Mutex<TileCtx<'d, T>>>,
    slots: Vec<Mutex<RowSlot<T>>>,
    order: Mutex<RowOrder<T>>,
    meta: Mutex<TileMeta>,
    /// First error in decoding order (keyed like `decode_tile_unit`).
    err: Mutex<Option<(usize, Error)>>,
    /// Tile row of every superblock row.
    tile_row: Vec<usize>,
    /// Deblocking arrays of the mode info (see `TileCtx::decode_row`).
    mi_arrays: (bool, bool, bool),
}

impl<'d, T: Pixel> TileJobs<'d, T> {
    fn finish_non_lr_row(
        &self,
        f: &FrameCtx,
        bp: &BandParams,
        rr: usize,
        rows: usize,
        mut bb: Box<BandBufs<T>>,
        pipe: &BandPipeline<T>,
    ) {
        let last = rr + 1 == rows;
        let (lum0, lum1) = band_rows(rr, last, f.sb_px);
        if bp.cdef {
            let m = self
                .meta
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            fill_band_meta(f, bp, &mut bb, rr, last, &m.cdef_idx, &m.lr);
        }
        bb.lf_v_done = true;
        pipe.push_band(BandJob {
            lum0,
            lum1,
            bufs: BandBuf::Owned(bb),
            bounds: Vec::new(),
        });
    }

    /// Row `r` has all its tiles: without LR, immediately hands off its
    /// bottom carry to `r + 1` and queues `r` (and `r + 1` if waiting)
    /// as soon as `r - 1` is ready; with LR, queues completed rows in
    /// order for `Finish`.
    fn complete(
        &self,
        f: &FrameCtx,
        bp: &BandParams,
        r: usize,
        mut bufs: Box<BandBufs<T>>,
        pipe: &BandPipeline<T>,
    ) {
        if !bp.lr {
            let (rows, ready_self, ready_next) = {
                let mut o = self
                    .order
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let rows = o.ready.len();
                let ready_self = if o.above_ready[r] {
                    if r > 0 {
                        if let Some(c) = o.carries[r - 1].take() {
                            apply_row_carry(f, bp, r, &c, &mut bufs);
                            o.free_carries.push(c);
                        }
                    }
                    true
                } else {
                    false
                };
                let ready_next = if r + 1 < rows {
                    o.above_ready[r + 1] = true;
                    if let Some(mut next_bb) = o.ready[r + 1].take() {
                        copy_row_carry_direct(f, bp, r + 1, &bufs, &mut next_bb);
                        Some((r + 1, next_bb))
                    } else {
                        let mut c = o.free_carries.pop().unwrap_or_else(|| RowCarry {
                            planes: std::mem::take(&mut bufs.carry_planes),
                            mi: std::mem::take(&mut bufs.carry_mi),
                        });
                        save_row_carry(f, bp, r, &bufs, &mut c);
                        o.carries[r] = Some(c);
                        None
                    }
                } else {
                    None
                };
                let ready_self = if ready_self {
                    Some((r, bufs))
                } else {
                    o.ready[r] = Some(bufs);
                    None
                };
                (rows, ready_self, ready_next)
            };
            if let Some((rr, bb)) = ready_self {
                self.finish_non_lr_row(f, bp, rr, rows, bb, pipe);
            }
            if let Some((rr, bb)) = ready_next {
                self.finish_non_lr_row(f, bp, rr, rows, bb, pipe);
            }
            return;
        }
        let mut o = self
            .order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        o.ready[r] = Some(bufs);
        let rows = o.ready.len();
        while o.next < rows {
            let rr = o.next;
            let Some(mut bb) = o.ready[rr].take() else {
                break;
            };
            let last = rr + 1 == rows;
            if bp.deblock || bp.cdef {
                let (m0, _) = band_mi_rows(f, rr);
                if rr > 0 {
                    copy_mi_rows(
                        &mut bb.mi,
                        &o.carry,
                        m0,
                        rr * f.sb4,
                        0,
                        f.mi_cols,
                        f.ss_x,
                        f.ss_y,
                    );
                }
                if !last {
                    let c0 = (rr + 1) * f.sb4 - 2;
                    let mi = &bb.mi;
                    o.carry.reset(
                        mi.stride,
                        2,
                        0,
                        c0,
                        f.ss_x,
                        f.ss_y,
                        !mi.lf_tx[0].is_empty(),
                        !mi.lf_tx[1].is_empty(),
                        !mi.delta_lf.is_empty(),
                        false,
                    );
                    copy_mi_rows(&mut o.carry, mi, c0, c0 + 2, 0, f.mi_cols, f.ss_x, f.ss_y);
                }
            }
            {
                let m = self
                    .meta
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                fill_band_meta(f, bp, &mut bb, rr, last, &m.cdef_idx, &m.lr);
            }
            pipe.push_row(RowJob {
                r: rr,
                last,
                bufs: bb,
            });
            o.next += 1;
        }
    }
}

impl<'d, T: Pixel> UnitRunner<T> for TileJobs<'d, T> {
    fn run_unit(
        &self,
        w: &mut BandWorker<'_, T>,
        bp: &BandParams,
        lane: usize,
        u: usize,
        pipe: &BandPipeline<T>,
    ) -> (bool, bool) {
        let f = &w.f;
        let ti = &f.fh.tile_info;
        let np = f.num_planes;
        let (sb4, sb_px) = (f.sb4, f.sb_px);
        let (r, tc) = (u / ti.cols, u % ti.cols);
        let tr = self.tile_row[r];
        let (mi_c0, mi_c1) = (ti.mi_col_starts[tc], ti.mi_col_starts[tc + 1]);
        // The tile's previous row is done (see `BandPipeline::set_units`),
        // so the lock is free.
        let mut ctx = self.ctxs[tr * ti.cols + tc]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let data = self.fd.tiles()[tr * ti.cols + tc];
        if let Err(e) = ctx.decode_row(f, &w.dsp, data, self.mi_arrays, bp.cdef, tr, tc, r) {
            let key = (r << 16) | (tr << 8) | tc;
            let mut err = self
                .err
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if err.as_ref().is_none_or(|(k, _)| key < *k) {
                *err = Some((key, e));
            }
            return (false, false);
        }
        let s = &ctx.s;
        // The frame's cdef_idx entries and restoration units coded in this
        // superblock row of the tile (before the row can complete).
        if bp.cdef || bp.lr {
            let mut m = self
                .meta
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if bp.cdef {
                let (c0, c1) = (mi_c0 / 16, mi_c1.div_ceil(16));
                let cs = bp.cdef_stride;
                for cr in r * sb4 / 16..((r + 1) * sb4).div_ceil(16) {
                    let src = (cr - s.cdef_row0) * s.cdef_stride + (c0 - s.cdef_col0);
                    m.cdef_idx[cr * cs + c0..cr * cs + c1]
                        .copy_from_slice(&s.cdef_idx[src..src + c1 - c0]);
                }
            }
            for p in (0..np).filter(|&p| bp.lr_on[p]) {
                let (sx, sy) = f.sub(p);
                let (dl, sl) = (&mut m.lr[p], &s.lr[p]);
                let us = sl.unit_size;
                // Spec 5.11.57 (read_lr): the units coded in this
                // superblock row of the tile.
                let ur0 = (r * sb4 * (MI_SIZE >> sy)).div_ceil(us);
                let ur1 = sl
                    .unit_rows
                    .min(((r + 1) * sb4 * (MI_SIZE >> sy)).div_ceil(us));
                let uc0 = (mi_c0 * (MI_SIZE >> sx)).div_ceil(us);
                let uc1 = sl
                    .unit_cols
                    .min((mi_c1.div_ceil(sb4) * sb4 * (MI_SIZE >> sx)).div_ceil(us));
                for ur in ur0..ur1 {
                    let b = ur * sl.unit_cols;
                    if uc0 < uc1 {
                        dl.units[b + uc0..b + uc1].copy_from_slice(&sl.units[b + uc0..b + uc1]);
                    }
                }
            }
        }
        let done = {
            let mut slot = self.slots[r]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let bb = slot.bufs.get_or_insert_with(|| {
                let mut bb = pipe.take_bufs();
                for p in 0..np {
                    let sy = f.sub(p).1;
                    let y0 = if r == 0 { 0 } else { (r * sb_px - BAND_TOP) >> sy };
                    bb.planes[p].reset(bp.out_stride[p], (sb_px + BAND_TOP) >> sy, 0, y0);
                }
                if bp.deblock || bp.cdef {
                    reset_band_mi(&mut bb.mi, &s.mi, f, r);
                }
                bb
            });
            for p in 0..np {
                let (sx, sy) = f.sub(p);
                let (x0, x1) = ((mi_c0 * 4) >> sx, (mi_c1 * 4) >> sx);
                let y1 = (((r + 1) * sb_px) >> sy).min(bp.mi_h[p]);
                for y in (r * sb_px) >> sy..y1 {
                    bb.planes[p].row_mut(y)[x0..x1].copy_from_slice(&s.planes[p].row(y)[..x1 - x0]);
                }
            }
            if bp.deblock || bp.cdef {
                copy_mi_rows(
                    &mut bb.mi,
                    &s.mi,
                    r * sb4,
                    ((r + 1) * sb4).min(f.mi_rows),
                    mi_c0,
                    mi_c1,
                    f.ss_x,
                    f.ss_y,
                );
            }
            slot.remaining -= 1;
            if slot.remaining == 0 {
                slot.bufs.take()
            } else {
                None
            }
        };
        drop(ctx);
        if let Some(mut bb) = done {
            // The tile's own state (`ctx`) is already updated for row
            // `r + 1`; release the lane before full-row deblocking so a
            // free worker can immediately advance this tile column.
            pipe.release_lane(lane);
            if bp.deblock {
                // All edges of the row but the horizontal ones on its top
                // boundary only touch the row itself (see
                // `deblock::deblock_rows_passes`): filter them here, in
                // parallel with the other rows, leaving only the top edges
                // to the ordered `finish_row` chain.
                deblock::deblock_rows_passes(
                    f,
                    &bb.mi,
                    &mut bb.planes[..np],
                    r * sb4,
                    (r + 1) * sb4,
                    bp.col_range,
                    &w.dsp,
                    &mut w.sc.lf_grid,
                    deblock::PASS_V | deblock::PASS_H_BODY,
                );
                bb.lf_v_done = true;
            }
            self.complete(f, bp, r, bb, pipe);
            return (true, true);
        }
        (true, false)
    }
}

/// Per-thread state of [`Pipeline::run_tiles_direct`], reused for all
/// tiles the thread decodes.
struct TileWorker<T: Pixel> {
    s: FrameState<T>,
    sc: Scratch,
    cdf: Option<Box<CdfContext>>,
}

impl<T: Pixel> TileWorker<T> {
    fn new() -> Self {
        Self {
            s: FrameState::default(),
            sc: Scratch::new(),
            cdf: None,
        }
    }

    /// Decodes tile `(tr, tc)` one superblock row at a time into
    /// `self.s.planes` (one superblock row high, the unfiltered last row
    /// is kept in `above_line` for the next row) and copies every row into
    /// the frame superblock row `chunks[r]` (planes with `strides`, rows
    /// below `mi_h` copied). Errors carry the key of the failing
    /// superblock row in decoding order.
    #[allow(clippy::too_many_arguments)]
    fn decode_tile(
        &mut self,
        f: &FrameCtx,
        dsp: &crate::decoder::av1::dsp::Dsp<T>,
        data: &[u8],
        tr: usize,
        tc: usize,
        chunks: &[Mutex<[&mut [T]; 3]>],
        strides: &[usize; 3],
        mi_h: &[usize; 3],
        min_failed_key: &std::sync::atomic::AtomicUsize,
    ) -> std::result::Result<(), (usize, Error)> {
        use std::sync::atomic::Ordering;
        let fh = f.fh;
        let ti = &fh.tile_info;
        let np = f.num_planes;
        let (sb4, sb_px) = (f.sb4, f.sb_px);
        let (mi_r0, mi_r1) = (ti.mi_row_starts[tr], ti.mi_row_starts[tr + 1]);
        let (mi_c0, mi_c1) = (ti.mi_col_starts[tc], ti.mi_col_starts[tc + 1]);
        let (sb_r0, sb_r1) = (mi_r0 / sb4, mi_r1.div_ceil(sb4));
        let tile_mi_al = (mi_c1 - mi_c0).div_ceil(sb4) * sb4;
        let s = &mut self.s;
        let mut x_range = [(0usize, 0usize); 3];
        for p in 0..np {
            let (sx, sy) = f.sub(p);
            // With intra block copy the window holds the whole tile.
            // `intrabc_mode_info` keeps the source inside the tile, so the
            // 2-tap chroma filter of `predict_intrabc` should not read past
            // it; the padding keeps a stray read from panicking anyway.
            let pad = if fh.allow_intrabc { 4 } else { 0 };
            let stride = (((tile_mi_al * 4) >> sx) + pad).next_multiple_of(64);
            x_range[p] = ((mi_c0 * 4) >> sx, (mi_c1 * 4) >> sx);
            let rows = if fh.allow_intrabc {
                (((sb_r1 - sb_r0) * sb_px) >> sy) + pad
            } else {
                sb_px >> sy
            };
            s.planes[p].reset(stride, rows, x_range[p].0, (sb_r0 * sb_px) >> sy);
            if fh.allow_intrabc {
                s.above_line[p].clear();
            } else {
                s.above_line[p].resize(stride + 64, T::default());
            }
        }
        s.mi.reset(
            tile_mi_al + 32,
            2 * sb4,
            mi_c0,
            (sb_r0 * sb4).saturating_sub(sb4),
            f.ss_x,
            f.ss_y,
            false,
            false,
            false,
            fh.allow_intrabc,
        );
        s.above
            .reset(mi_c0, tile_mi_al + 32, np, fh.allow_screen_content_tools);
        s.cdef_idx.clear();
        (s.cdef_stride, s.cdef_row0, s.cdef_col0) = (0, 0, 0);
        for lp in s.lr.iter_mut() {
            lp.units.clear();
        }
        s.ipred = [IntraSetup::new(f, 0), IntraSetup::new(f, 1)];
        let mut ts = TileState::with_cdf(
            data,
            fh,
            f.seq,
            (mi_r0, mi_r1),
            (mi_c0, mi_c1),
            self.cdf.take(),
        );
        let mut res = Ok(());
        for r in sb_r0..sb_r1 {
            let key = (r << 16) | (tr << 8) | tc;
            if key > min_failed_key.load(Ordering::Relaxed) {
                // An earlier superblock row failed: its error is reported.
                res = Err((usize::MAX, Error::Truncated));
                break;
            }
            let mi_row = r * sb4;
            let sb_y = r * sb_px;
            s.mi.slide_to(mi_row.saturating_sub(sb4));
            if !fh.allow_intrabc {
                for p in 0..np {
                    s.planes[p].y0 = sb_y >> f.sub(p).1;
                }
            }
            s.left.clear();
            s.set_sb_y(f, sb_y);
            let mut dec = SbDecoder {
                f,
                t: &mut ts,
                s: &mut *s,
                sc: &mut self.sc,
                dsp,
                b: Default::default(),
                recon: true,
                sb_y,
                read_deltas: false,
                plane_tx_type: crate::decoder::av1::types::TxType::DCT_DCT,
                stage: stage::DecoderStage::Fused,
                filter_type: [false; 2],
            };
            let mut c = mi_c0;
            while c < mi_c1 {
                if let Err(e) = dec.decode_sb(mi_row, c) {
                    res = Err((key, e));
                    break;
                }
                if dec.t.msac.overrun() {
                    res = Err((key, Error::Truncated));
                    break;
                }
                c += sb4;
            }
            if res.is_err() {
                break;
            }
            if !fh.allow_intrabc && r + 1 < sb_r1 {
                for p in 0..np {
                    let h = sb_px >> f.sub(p).1;
                    let w = x_range[p].1 - x_range[p].0;
                    let pl = &s.planes[p];
                    let off = (h - 1) * pl.stride;
                    s.above_line[p][..w].copy_from_slice(&pl.data()[off..off + w]);
                }
            }
            {
                let mut dst = chunks[r]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for p in 0..np {
                    let sy = f.sub(p).1;
                    let (x0, x1) = x_range[p];
                    let w = x1 - x0;
                    let y0 = sb_y >> sy;
                    let h = ((sb_y + sb_px) >> sy).min(mi_h[p]).saturating_sub(y0);
                    let pl = &s.planes[p];
                    let src_stride = pl.stride;
                    let src_data = pl.data();
                    let sr0 = y0 - pl.y0;
                    let dst_stride = strides[p];
                    let dst_data = &mut dst[p][x0..];
                    for ry in 0..h {
                        let d = ry * dst_stride;
                        let sr = (sr0 + ry) * src_stride;
                        dst_data[d..d + w].copy_from_slice(&src_data[sr..sr + w]);
                    }
                }
            }
        }
        self.cdf = Some(ts.cdf);
        res
    }
}

#[allow(unused)]
fn unsupported() -> Error {
    Error::Unsupported("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::{decode, DecodeOptions};

    // Minimal 150x150 (3 superblock rows) 8-bit AV1 keyframe OBU stream.
    const OBU_150X150: [u8; 39] = [
        18, 0, 10, 14, 0, 0, 0, 3, 188, 172, 169, 181, 242, 32, 33, 160, 208, 128, 50, 19, 16, 0,
        131, 128, 0, 0, 128, 0, 0, 0, 235, 197, 166, 46, 12, 13, 209, 81, 64,
    ];

    #[test]
    fn multithreaded_thread_counts_match_single_threaded() {
        let ref_frame = decode(
            &OBU_150X150,
            &DecodeOptions {
                threads: 1,
                ..Default::default()
            },
        )
        .expect("single-threaded decode");
        for threads in [2, 3, 4, 6] {
            for _ in 0..8 {
                let got = decode(
                    &OBU_150X150,
                    &DecodeOptions {
                        threads,
                        ..Default::default()
                    },
                )
                .expect("multithreaded decode");
                assert_eq!(got, ref_frame, "mismatch for threads={threads}");
            }
        }
    }

    #[test]
    fn handoff_guard_marks_terminal_state() {
        for initial in [0u8, 1] {
            let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(initial));
            drop(HandoffGuard(state.clone()));
            assert_eq!(state.load(std::sync::atomic::Ordering::Acquire), 3);
        }
        let handed_off = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(2));
        drop(HandoffGuard(handed_off.clone()));
        assert_eq!(handed_off.load(std::sync::atomic::Ordering::Acquire), 2);
    }

    #[test]
    fn work_spin_drains_final_row_pushed_before_close() {
        let fd = crate::decoder::av1::obu::parse_first_frame(&OBU_150X150).unwrap();
        let stride = 192usize;
        let sb_px = 64usize;
        let bp = BandParams {
            deblock: false,
            cdef: false,
            lr: false,
            lr_on: [false; 3],
            mi_w: [152, 76, 76],
            mi_h: [152, 76, 76],
            out_w: [150, 75, 75],
            out_h: [150, 75, 75],
            out_stride: [stride, stride, stride],
            cdef_stride: 0,
            frame_h: 150,
            col_range: (0, 38),
        };
        for _ in 0..64 {
            let out_planes = [
                PlaneWin::from_vec(vec![0u8; stride * 192], stride, 192),
                PlaneWin::from_vec(vec![0u8; stride * 96], stride, 96),
                PlaneWin::from_vec(vec![0u8; stride * 96], stride, 96),
            ];
            let pipe = BandPipeline::<u8>::with_pooled(
                out_planes,
                Vec::new(),
                Default::default(),
                Vec::new(),
                Vec::new(),
                0,
            );
            std::thread::scope(|scope| {
                let pipe_ref = &pipe;
                let prod = scope.spawn(move || {
                    std::thread::yield_now();
                    for r in 0..3 {
                        let last = r == 2;
                        let mut bb = Box::<BandBufs<u8>>::default();
                        for p in 0..3 {
                            let sy = usize::from(p > 0);
                            let y0 = if r == 0 { 0 } else { (r * sb_px - BAND_TOP) >> sy };
                            let rows = if r == 0 { sb_px >> sy } else { (sb_px + BAND_TOP) >> sy };
                            bb.planes[p].reset(stride, rows, 0, y0);
                            bb.planes[p].data_mut().fill((r as u8 + 1) * 10 + p as u8);
                        }
                        pipe_ref.push_row(RowJob { r, last, bufs: bb });
                    }
                    pipe_ref.close();
                });
                let mut w = BandWorker::<u8>::new(&fd);
                pipe.work_spin(&mut w, &bp, None, || prod.is_finished());
                drop(prod);
            });
            let mut free = Vec::new();
            let mut carry = Default::default();
            let mut lf = Vec::new();
            let mut lr_free = Vec::new();
            let mut scs = Vec::new();
            let out = pipe.reclaim_into(&mut free, &mut carry, &mut lf, &mut lr_free, &mut scs);
            let out_y = out[0].data();
            // Every row including the final superblock row (y = 149) must be written.
            assert_eq!(out_y[0], 10);
            assert_eq!(out_y[149 * stride], 30);
            assert_eq!(out_y[149 * stride + 149], 30);
        }
    }

    #[test]
    fn scoped_worker_panic_aborts_without_panicking_scope() {
        let fd = crate::decoder::av1::obu::parse_first_frame(&OBU_150X150).unwrap();
        let bp = BandParams {
            deblock: false,
            cdef: false,
            lr: false,
            lr_on: [false; 3],
            mi_w: [64, 32, 32],
            mi_h: [64, 32, 32],
            out_w: [64, 32, 32],
            out_h: [64, 32, 32],
            out_stride: [64, 64, 64],
            cdef_stride: 0,
            frame_h: 64,
            col_range: (0, 16),
        };
        let pipe = BandPipeline::<u8>::with_pooled(
            Default::default(),
            Vec::new(),
            Default::default(),
            Vec::new(),
            Vec::new(),
            1,
        );
        std::thread::scope(|scope| {
            let pipe_ref = &pipe;
            let h = scope.spawn(move || {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _abort = AbortOnPanic(pipe_ref);
                    panic!("simulated worker panic");
                }));
            });
            let mut w = BandWorker::<u8>::new(&fd);
            pipe.work_spin(&mut w, &bp, None, || h.is_finished());
            drop(h);
        });
        assert!(pipe.is_aborted());
    }
}
