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

//! Film grain synthesis (spec 7.18.3).
//!
//! OWNER: film grain worker. Contract below must be kept (the struct
//! `FilmGrainParams` is filled by the header parser; do not rename
//! fields).
//!
//! # Implementation notes
//!
//! The spec describes grain synthesis with full-frame `noiseStripe` and
//! `noiseImage` arrays. This implementation instead works one 32x32 luma
//! block (and the co-located chroma block) at a time, like dav1d: the noise
//! of a sample only depends on the grain template offsets of its own block
//! and, when `overlap_flag` is set, of the blocks to the left, above and
//! above-left (whose last rows / columns are blended in). The offsets of
//! every block are derived from a random number generator that is reseeded
//! for every row of blocks, so any row of blocks (and therefore any band
//! starting at a multiple of 32 luma rows) can be processed independently
//! without any state carried over from the previous band.
//!
//! Equivalence with the spec, for a sample at column `c` (plane units, block
//! width `bw = 32 >> sub_x`): in the spec, block `k` writes stripe columns
//! `k * bw .. k * bw + bw + overlap` and the following block overwrites (or,
//! with `overlap_flag`, blends into) the trailing `overlap` columns. Hence
//! the stripe value at column `c` is the grain of block `c / bw` at offset
//! `c % bw`, blended with the grain of block `c / bw - 1` at offset
//! `c % bw + bw` when `c % bw < overlap`. The same holds vertically for the
//! stripes. The horizontal blend is done first, then the vertical one, with
//! clipping after each step, exactly as in the spec.

use crate::decoder::av1::pixel::{clip3, round2, Pixel};

/// Film grain parameters, field names as in spec 5.9.30 / 6.8.20.
/// Arrays are sized to their maximum and only the first `num_*` entries are
/// meaningful.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub num_y_points: usize,
    pub point_y_value: [u8; 16],
    pub point_y_scaling: [u8; 16],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: usize,
    pub point_cb_value: [u8; 16],
    pub point_cb_scaling: [u8; 16],
    pub num_cr_points: usize,
    pub point_cr_value: [u8; 16],
    pub point_cr_scaling: [u8; 16],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// Height of the luma grain template.
pub const GRAIN_H: usize = 73;
/// Width of the luma grain template.
pub const GRAIN_W: usize = 82;

/// A grain template (`LumaGrain`, `CbGrain` or `CrGrain`). Chroma templates
/// only use the top-left `chromaH x chromaW` part.
pub type GrainTemplate = [[i16; GRAIN_W]; GRAIN_H];

/// Size of a grain block in luma samples.
pub(crate) const BLOCK: usize = 32;

/// The spec's 16-bit linear feedback shift register (7.18.3.2).
#[derive(Clone, Copy, Debug)]
struct Rng(u16);

impl Rng {
    /// `get_random_number(bits)`.
    #[inline]
    fn next(&mut self, bits: u32) -> i32 {
        let r = self.0 as u32;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        let r = (r >> 1) | (bit << 15);
        self.0 = r as u16;
        ((r >> (16 - bits)) & ((1 << bits) - 1)) as i32
    }

    /// Generator for the block row `luma_num` (start of each iteration of
    /// the outer loop of the noise stripe construction in 7.18.3.5).
    fn for_block_row(seed: u16, luma_num: usize) -> Self {
        let n = luma_num as u32;
        let mut r = seed as u32;
        r ^= ((n * 37 + 178) & 255) << 8;
        r ^= (n * 173 + 105) & 255;
        Rng(r as u16)
    }
}

/// Per chroma plane parameters for the `merged` computation of 7.18.3.5.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChromaParams {
    pub mult: i32,
    pub luma_mult: i32,
    /// `(offset - 256) << (BitDepth - 8)`.
    pub offset: i32,
}

/// Grain template offsets of one block, in template samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockOffsets {
    pub x: usize,
    pub y: usize,
}

impl BlockOffsets {
    /// `planeOffsetX` / `planeOffsetY` for a block whose random value is
    /// `rand`.
    #[inline]
    pub(crate) fn new(rand: i32, sub_x: bool, sub_y: bool) -> Self {
        let ox = (rand >> 4) as usize;
        let oy = (rand & 15) as usize;
        BlockOffsets {
            x: if sub_x { 6 + ox } else { 9 + ox * 2 },
            y: if sub_y { 6 + oy } else { 9 + oy * 2 },
        }
    }
}

/// Offsets of a block and of its neighbours that are blended in. The
/// neighbours are `None` when there is no overlap with them (left or top
/// frame edge, or `overlap_flag == 0`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockNeighbourhood {
    pub cur: BlockOffsets,
    pub left: Option<BlockOffsets>,
    pub top: Option<BlockOffsets>,
    pub top_left: Option<BlockOffsets>,
}

/// Frame-level film grain state: grain templates and scaling lookup tables
/// (spec 7.18.3.3 and 7.18.3.4), computed once per frame.
pub struct FilmGrain {
    bd: u32,
    num_planes: usize,
    sub_x: bool,
    sub_y: bool,
    grain_seed: u16,
    overlap: bool,
    /// Whether noise is applied to each plane.
    apply: [bool; 3],
    chroma_scaling_from_luma: bool,
    chroma: [ChromaParams; 2],
    scaling_shift: u32,
    min_value: i32,
    max_luma: i32,
    max_chroma: i32,
    grain_min: i32,
    grain_max: i32,
    /// `LumaGrain`, `CbGrain`, `CrGrain`.
    grain: [Box<GrainTemplate>; 3],
    /// `scale_lut(plane, index)` for every index in `0 .. 1 << bd` (the
    /// table always has 4096 entries so that `index & 4095` needs no
    /// bounds check).
    scale: [Box<ScaleLut>; 3],
}

impl FilmGrain {
    /// `mono` is `mono_chrome`, `sub_x` / `sub_y` are the chroma subsampling
    /// flags and `mc_identity` is `matrix_coefficients == MC_IDENTITY`.
    pub fn new(
        params: &FilmGrainParams,
        bd: u32,
        mono: bool,
        sub_x: bool,
        sub_y: bool,
        mc_identity: bool,
    ) -> Self {
        let num_planes = if mono { 1 } else { 3 };
        let grain_center = 128i32 << (bd - 8);
        let grain_min = -grain_center;
        let grain_max = (256i32 << (bd - 8)) - 1 - grain_center;

        let grain = generate_grain(params, bd, mono, sub_x, sub_y);

        let csfl = params.chroma_scaling_from_luma && !mono;
        let mut scale = [
            Box::new([0; 4096]),
            Box::new([0; 4096]),
            Box::new([0; 4096]),
        ];
        for (plane, s) in scale.iter_mut().enumerate().take(num_planes) {
            let lut = scaling_lut(params, plane, csfl);
            expand_scaling_lut(&lut, bd, s);
        }

        let apply = [
            params.num_y_points > 0,
            !mono && (params.num_cb_points > 0 || csfl),
            !mono && (params.num_cr_points > 0 || csfl),
        ];

        let (min_value, max_luma, max_chroma) = if params.clip_to_restricted_range {
            let max_luma = 235 << (bd - 8);
            (
                16 << (bd - 8),
                max_luma,
                if mc_identity { max_luma } else { 240 << (bd - 8) },
            )
        } else {
            let max = (256 << (bd - 8)) - 1;
            (0, max, max)
        };

        let chroma_params = |mult: u8, luma_mult: u8, offset: u16| ChromaParams {
            mult: mult as i32 - 128,
            luma_mult: luma_mult as i32 - 128,
            offset: (offset as i32 - 256) * (1 << (bd - 8)),
        };

        FilmGrain {
            bd,
            num_planes,
            sub_x,
            sub_y,
            grain_seed: params.grain_seed,
            overlap: params.overlap_flag,
            apply,
            chroma_scaling_from_luma: csfl,
            chroma: [
                chroma_params(params.cb_mult, params.cb_luma_mult, params.cb_offset),
                chroma_params(params.cr_mult, params.cr_luma_mult, params.cr_offset),
            ],
            scaling_shift: params.grain_scaling_minus_8 as u32 + 8,
            min_value,
            max_luma,
            max_chroma,
            grain_min,
            grain_max,
            grain,
            scale,
        }
    }

    /// The grain template of `plane` (`LumaGrain`, `CbGrain`, `CrGrain`).
    pub fn grain_template(&self, plane: usize) -> &GrainTemplate {
        &self.grain[plane]
    }

    /// Applies grain (spec 7.18.3.5) in place to a horizontal band of the
    /// output frame.
    ///
    /// * The band covers luma rows `y0 .. y0 + band_h` of a frame of size
    ///   `frame_w x frame_h` (luma samples). `y0` is a multiple of 32 and
    ///   `band_h` is a multiple of 32 unless the band reaches `frame_h`.
    /// * `planes[p]` holds the band for plane `p` (1 entry for monochrome,
    ///   else 3); row 0 of `planes[p]` is luma row `y0` (chroma row
    ///   `y0 >> sub_y`) and rows are `strides[p]` apart. Every row holds the
    ///   full plane width.
    /// * Chroma noise must be computed from the luma samples *before* luma
    ///   noise is added.
    /// * Applying the function band by band must give the same result as
    ///   applying spec 7.18.3.5 to the whole frame.
    pub fn apply_band<T: Pixel>(
        &self,
        planes: &mut [&mut [T]],
        strides: &[usize],
        y0: usize,
        band_h: usize,
        frame_w: usize,
        frame_h: usize,
    ) {
        assert!(
            y0.is_multiple_of(BLOCK),
            "film grain bands must start at a multiple of 32 rows"
        );
        let y1 = (y0 + band_h).min(frame_h);
        assert!(
            y1 == frame_h || y1.is_multiple_of(BLOCK),
            "film grain band must end at a multiple of 32 rows"
        );
        if y0 >= y1 || frame_w == 0 || !self.apply.iter().any(|&a| a) {
            return;
        }
        assert!(planes.len() >= self.num_planes && strides.len() >= self.num_planes);

        let blocks_x = frame_w.div_ceil(BLOCK);
        let (luma, chroma) = planes.split_at_mut(1);
        let luma = &mut *luma[0];

        for by in y0 / BLOCK..y1.div_ceil(BLOCK) {
            let mut rng = Rng::for_block_row(self.grain_seed, by);
            // Generator of the row above, only needed for vertical overlap.
            let mut rng_top =
                (self.overlap && by > 0).then(|| Rng::for_block_row(self.grain_seed, by - 1));
            let mut prev: Option<(i32, Option<i32>)> = None;
            for bx in 0..blocks_x {
                let rand = rng.next(8);
                let rand_top = rng_top.as_mut().map(|r| r.next(8));
                let left = if self.overlap { prev } else { None };
                prev = Some((rand, rand_top));
                let rands = BlockRands {
                    cur: rand,
                    top: rand_top,
                    left: left.map(|l| l.0),
                    top_left: left.and_then(|l| l.1),
                };

                // Chroma first: it reads the noise-free luma samples of the
                // same block.
                for (ci, plane) in chroma.iter_mut().enumerate().take(self.num_planes - 1) {
                    if self.apply[ci + 1] {
                        self.chroma_block(
                            ci,
                            plane,
                            strides[ci + 1],
                            luma,
                            strides[0],
                            &rands,
                            bx,
                            by,
                            y0,
                            frame_w,
                            frame_h,
                        );
                    }
                }
                if self.apply[0] {
                    self.luma_block(luma, strides[0], &rands, bx, by, y0, frame_w, frame_h);
                }
            }
        }
    }

    /// Computes the noise of the visible part (`w x h` plane samples) of
    /// one block of `plane` into `out`.
    fn block_noise(
        &self,
        plane: usize,
        rands: &BlockRands,
        w: usize,
        h: usize,
        out: &mut [[i16; BLOCK]; BLOCK],
    ) {
        let (sx, sy) = if plane > 0 { (self.sub_x, self.sub_y) } else { (false, false) };
        let nb = rands.neighbourhood(sx, sy);
        noise_block(
            &self.grain[plane],
            &nb,
            sx,
            sy,
            w,
            h,
            self.grain_min,
            self.grain_max,
            out,
        );
    }

    /// Applies noise to one 32x32 luma block.
    #[allow(clippy::too_many_arguments)]
    fn luma_block<T: Pixel>(
        &self,
        luma: &mut [T],
        stride: usize,
        rands: &BlockRands,
        bx: usize,
        by: usize,
        y0: usize,
        frame_w: usize,
        frame_h: usize,
    ) {
        let x0 = bx * BLOCK;
        let yb = by * BLOCK;
        let w = BLOCK.min(frame_w - x0);
        let h = BLOCK.min(frame_h - yb);
        let p = NoiseParams {
            lut: &self.scale[0],
            scaling_shift: self.scaling_shift,
            min: self.min_value,
            max: self.max_luma,
        };
        let mut noise = [[0i16; BLOCK]; BLOCK];
        self.block_noise(0, rands, w, h, &mut noise);
        for (i, noise_row) in noise.iter().enumerate().take(h) {
            let start = (yb - y0 + i) * stride + x0;
            let row = &mut luma[start..start + w];
            luma_row(row, noise_row, &p);
        }
    }

    /// Applies noise to the chroma block co-located with luma block
    /// `(bx, by)` of chroma plane `ci + 1`.
    #[allow(clippy::too_many_arguments)]
    fn chroma_block<T: Pixel>(
        &self,
        ci: usize,
        plane: &mut [T],
        stride: usize,
        luma: &[T],
        luma_stride: usize,
        rands: &BlockRands,
        bx: usize,
        by: usize,
        y0: usize,
        frame_w: usize,
        frame_h: usize,
    ) {
        let (sx, sy) = (self.sub_x as usize, self.sub_y as usize);
        let bw = BLOCK >> sx;
        let bh = BLOCK >> sy;
        let plane_w = (frame_w + sx) >> sx;
        let plane_h = (frame_h + sy) >> sy;
        let x0 = bx * bw;
        let yb = by * bh;
        let w = bw.min(plane_w - x0);
        let h = bh.min(plane_h - yb);

        let p = ChromaNoiseParams {
            noise: NoiseParams {
                lut: &self.scale[ci + 1],
                scaling_shift: self.scaling_shift,
                min: self.min_value,
                max: self.max_chroma,
            },
            csfl: self.chroma_scaling_from_luma,
            cp: self.chroma[ci],
            pixel_max: (1 << self.bd) - 1,
            sx: sx == 1,
        };
        let band_y0 = y0 >> sy;
        let mut noise = [[0i16; BLOCK]; BLOCK];
        self.block_noise(ci + 1, rands, w, h, &mut noise);
        let lx0 = x0 << sx;
        for (i, noise_row) in noise.iter().enumerate().take(h) {
            let y = yb + i;
            let row_start = ((y << sy) - y0) * luma_stride;
            let luma_row = &luma[row_start + lx0..row_start + frame_w];
            let start = (y - band_y0) * stride + x0;
            let row = &mut plane[start..start + w];
            chroma_row(row, noise_row, luma_row, &p);
        }
    }
}

/// Parameters of the noise application (7.18.3.5) of one plane.
pub(crate) struct NoiseParams<'a> {
    /// `scale_lut(plane, index)` for every index.
    pub lut: &'a ScaleLut,
    /// `ScalingShift`.
    pub scaling_shift: u32,
    /// `minValue`.
    pub min: i32,
    /// `maxLuma` or `maxChroma`.
    pub max: i32,
}

impl NoiseParams<'_> {
    /// Adds the scaled `noise` to the sample `orig` whose scaling function
    /// index is `index`, and clips the result.
    #[inline(always)]
    fn apply(&self, orig: i32, index: i32, noise: i16) -> i32 {
        let v = round2(
            self.lut[index as usize & 4095] as i32 * noise as i32,
            self.scaling_shift,
        );
        clip3(self.min, self.max, orig + v)
    }
}

/// Parameters of the noise application of one chroma plane.
pub(crate) struct ChromaNoiseParams<'a> {
    pub noise: NoiseParams<'a>,
    /// `chroma_scaling_from_luma`.
    pub csfl: bool,
    pub cp: ChromaParams,
    /// `(1 << BitDepth) - 1`.
    pub pixel_max: i32,
    /// `subsampling_x`.
    pub sx: bool,
}

impl ChromaNoiseParams<'_> {
    /// `merged` of 7.18.3.5 for the chroma sample `orig` with
    /// `averageLuma == avg`.
    #[inline(always)]
    fn merged(&self, avg: i32, orig: i32) -> i32 {
        if self.csfl {
            avg
        } else {
            let combined = avg * self.cp.luma_mult + orig * self.cp.mult;
            clip3(0, self.pixel_max, (combined >> 6) + self.cp.offset)
        }
    }
}

/// Adds noise to one row of luma samples: `row[j]` gets `noise[j]`
/// scaled by the scaling function of its own value. This is the kernel
/// a SIMD version replaces.
pub(crate) fn luma_row<T: Pixel>(row: &mut [T], noise: &[i16], p: &NoiseParams) {
    for (px, &n) in row.iter_mut().zip(noise) {
        let orig = px.to_i32();
        *px = T::from_i32(p.apply(orig, orig, n));
    }
}

/// Adds noise to one row of chroma samples. `luma` is the co-located
/// noise-free luma row from the first luma sample of the row to the
/// right frame edge (`averageLuma` uses the next luma sample of a pair,
/// or the last one of the row at the right edge of an odd width frame).
/// This is the kernel a SIMD version replaces.
pub(crate) fn chroma_row<T: Pixel>(
    row: &mut [T],
    noise: &[i16],
    luma: &[T],
    p: &ChromaNoiseParams,
) {
    let w = row.len();
    // averageLuma of every sample of the row.
    let mut avg = [0i32; BLOCK];
    if p.sx {
        if 2 * w <= luma.len() {
            for (a, &[l0, l1]) in avg.iter_mut().zip(luma[..2 * w].as_chunks::<2>().0) {
                *a = (l0.to_i32() + l1.to_i32() + 1) >> 1;
            }
        } else {
            // Right frame edge with an odd width.
            for (j, a) in avg[..w].iter_mut().enumerate() {
                let next = (2 * j + 1).min(luma.len() - 1);
                *a = (luma[2 * j].to_i32() + luma[next].to_i32() + 1) >> 1;
            }
        }
    } else {
        for (a, l) in avg.iter_mut().zip(&luma[..w]) {
            *a = l.to_i32();
        }
    }
    for ((px, &n), &average_luma) in row.iter_mut().zip(noise).zip(&avg) {
        let orig = px.to_i32();
        *px = T::from_i32(p.noise.apply(orig, p.merged(average_luma, orig), n));
    }
}

/// Random values (`rand` of 7.18.3.5) of a block and of its neighbours.
#[derive(Clone, Copy, Debug)]
struct BlockRands {
    cur: i32,
    left: Option<i32>,
    top: Option<i32>,
    top_left: Option<i32>,
}

impl BlockRands {
    fn neighbourhood(&self, sx: bool, sy: bool) -> BlockNeighbourhood {
        let o = |r: i32| BlockOffsets::new(r, sx, sy);
        BlockNeighbourhood {
            cur: o(self.cur),
            left: self.left.map(o),
            top: self.top.map(o),
            top_left: self.top_left.map(o),
        }
    }
}

/// Blends `old` (the sample of the previous block) with `new` at position
/// `idx` of an overlap region of `len` samples (2 without subsampling, 1
/// with subsampling) and clips the result.
#[inline(always)]
pub(crate) fn blend(old: i32, new: i32, idx: usize, len: usize, gmin: i32, gmax: i32) -> i32 {
    let g = if len == 1 {
        old * 23 + new * 22
    } else if idx == 0 {
        old * 27 + new * 17
    } else {
        old * 17 + new * 27
    };
    clip3(gmin, gmax, round2(g, 5))
}

/// Computes the noise of the top-left `w x h` samples of a block (plane
/// units), following the noise stripe / noise image construction of
/// 7.18.3.5. This is the kernel a SIMD version would replace.
#[allow(clippy::too_many_arguments)]
pub(crate) fn noise_block(
    grain: &GrainTemplate,
    nb: &BlockNeighbourhood,
    sx: bool,
    sy: bool,
    w: usize,
    h: usize,
    gmin: i32,
    gmax: i32,
    out: &mut [[i16; BLOCK]; BLOCK],
) {
    let bw = BLOCK >> sx as usize;
    let bh = BLOCK >> sy as usize;
    let ow = if sx { 1 } else { 2 };
    let oh = if sy { 1 } else { 2 };

    // Horizontally blended stripe row `row` (0 .. bh + oh) of the block
    // with offsets `cur` and left neighbour `left`, columns 0..w.
    let stripe =
        |cur: BlockOffsets, left: Option<BlockOffsets>, row: usize, out: &mut [i16; BLOCK]| {
            out[..w].copy_from_slice(&grain[cur.y + row][cur.x..cur.x + w]);
            if let Some(l) = left {
                let left_row = &grain[l.y + row];
                for j in 0..ow.min(w) {
                    let old = left_row[l.x + j + bw] as i32;
                    out[j] = blend(old, out[j] as i32, j, ow, gmin, gmax) as i16;
                }
            }
        };

    for (i, out_row) in out.iter_mut().enumerate().take(h) {
        stripe(nb.cur, nb.left, i, out_row);
        if let Some(top) = nb.top {
            if i < oh {
                let mut old = [0i16; BLOCK];
                stripe(top, nb.top_left, i + bh, &mut old);
                for (o, &t) in out_row[..w].iter_mut().zip(&old[..w]) {
                    *o = blend(t as i32, *o as i32, i, oh, gmin, gmax) as i16;
                }
            }
        }
    }
}

/// Generate grain process (7.18.3.3). Returns `[LumaGrain, CbGrain,
/// CrGrain]`; chroma templates are all zero for monochrome.
fn generate_grain(
    p: &FilmGrainParams,
    bd: u32,
    mono: bool,
    sub_x: bool,
    sub_y: bool,
) -> [Box<GrainTemplate>; 3] {
    let grain_center = 128i32 << (bd - 8);
    let gmin = -grain_center;
    let gmax = (256i32 << (bd - 8)) - 1 - grain_center;
    let shift = 12 - bd + p.grain_scale_shift as u32;
    let ar_shift = p.ar_coeff_shift_minus_6 as u32 + 6;
    let lag = p.ar_coeff_lag as isize;

    let mut luma = Box::new([[0i16; GRAIN_W]; GRAIN_H]);
    let mut rng = Rng(p.grain_seed);
    if p.num_y_points > 0 {
        for row in luma.iter_mut() {
            for v in row.iter_mut() {
                *v = round2(GAUSSIAN_SEQUENCE[rng.next(11) as usize] as i32, shift) as i16;
            }
        }
        // Auto-regressive filter (skipped when there is no luma noise: the
        // template is all zero and stays so).
        for y in 3..GRAIN_H {
            for x in 3..GRAIN_W - 3 {
                let mut s = 0i32;
                let mut pos = 0;
                'outer: for dr in -lag..=0 {
                    for dc in -lag..=lag {
                        if dr == 0 && dc == 0 {
                            break 'outer;
                        }
                        let c = p.ar_coeffs_y_plus_128[pos] as i32 - 128;
                        s +=
                            luma[(y as isize + dr) as usize][(x as isize + dc) as usize] as i32 * c;
                        pos += 1;
                    }
                }
                luma[y][x] = clip3(gmin, gmax, luma[y][x] as i32 + round2(s, ar_shift)) as i16;
            }
        }
    }

    let mut cb = Box::new([[0i16; GRAIN_W]; GRAIN_H]);
    let mut cr = Box::new([[0i16; GRAIN_W]; GRAIN_H]);
    if !mono {
        let chroma_w = if sub_x { 44 } else { 82 };
        let chroma_h = if sub_y { 38 } else { 73 };
        let csfl = p.chroma_scaling_from_luma;
        let fill = |t: &mut GrainTemplate, enabled: bool, seed: u16| {
            if !enabled {
                return;
            }
            let mut rng = Rng(seed);
            for row in t.iter_mut().take(chroma_h) {
                for v in row.iter_mut().take(chroma_w) {
                    *v = round2(GAUSSIAN_SEQUENCE[rng.next(11) as usize] as i32, shift) as i16;
                }
            }
        };
        fill(&mut cb, p.num_cb_points > 0 || csfl, p.grain_seed ^ 0xb524);
        fill(&mut cr, p.num_cr_points > 0 || csfl, p.grain_seed ^ 0x49d8);

        let (sxu, syu) = (sub_x as usize, sub_y as usize);
        for y in 3..chroma_h {
            for x in 3..chroma_w - 3 {
                let mut s0 = 0i32;
                let mut s1 = 0i32;
                let mut pos = 0;
                'outer2: for dr in -lag..=0 {
                    for dc in -lag..=lag {
                        let c0 = p.ar_coeffs_cb_plus_128[pos] as i32 - 128;
                        let c1 = p.ar_coeffs_cr_plus_128[pos] as i32 - 128;
                        if dr == 0 && dc == 0 {
                            if p.num_y_points > 0 {
                                let luma_x = ((x - 3) << sxu) + 3;
                                let luma_y = ((y - 3) << syu) + 3;
                                let mut l = 0i32;
                                for i in 0..=syu {
                                    for j in 0..=sxu {
                                        l += luma[luma_y + i][luma_x + j] as i32;
                                    }
                                }
                                let l = round2(l, (sxu + syu) as u32);
                                s0 += l * c0;
                                s1 += l * c1;
                            }
                            break 'outer2;
                        }
                        let yy = (y as isize + dr) as usize;
                        let xx = (x as isize + dc) as usize;
                        s0 += cb[yy][xx] as i32 * c0;
                        s1 += cr[yy][xx] as i32 * c1;
                        pos += 1;
                    }
                }
                cb[y][x] = clip3(gmin, gmax, cb[y][x] as i32 + round2(s0, ar_shift)) as i16;
                cr[y][x] = clip3(gmin, gmax, cr[y][x] as i32 + round2(s1, ar_shift)) as i16;
            }
        }
    }
    [luma, cb, cr]
}

/// Scaling lookup initialization process (7.18.3.4) for one plane.
fn scaling_lut(p: &FilmGrainParams, plane: usize, csfl: bool) -> [u8; 256] {
    let (n, xs, ys) = if plane == 0 || csfl {
        (p.num_y_points, &p.point_y_value, &p.point_y_scaling)
    } else if plane == 1 {
        (p.num_cb_points, &p.point_cb_value, &p.point_cb_scaling)
    } else {
        (p.num_cr_points, &p.point_cr_value, &p.point_cr_scaling)
    };
    let mut lut = [0u8; 256];
    if n == 0 {
        return lut;
    }
    let n = n.min(xs.len());
    lut[..xs[0] as usize].fill(ys[0]);
    for i in 0..n - 1 {
        let (x0, x1) = (xs[i] as i32, xs[i + 1] as i32);
        let y0 = ys[i] as i32;
        let delta_y = ys[i + 1] as i32 - y0;
        let delta_x = x1 - x0;
        if delta_x <= 0 {
            // Non-conformant (points must be increasing); ignore segment.
            continue;
        }
        let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
        for x in 0..delta_x {
            let v = y0 + ((x * delta + 32768) >> 16);
            lut[(x0 + x) as usize] = v as u8;
        }
    }
    lut[xs[n - 1] as usize..].fill(ys[n - 1]);
    lut
}

/// `scale_lut(plane, index)` for all indices `0 .. 4096`.
pub(crate) type ScaleLut = [u8; 4096];

/// Precomputes `scale_lut(plane, index)` (7.18.3.5) for all indices
/// `0 .. 1 << bd` (entries beyond are left untouched).
fn expand_scaling_lut(lut: &[u8; 256], bd: u32, out: &mut ScaleLut) {
    let shift = bd - 8;
    let values = (0..1usize << bd).map(|index| {
        let x = index >> shift;
        let rem = (index - (x << shift)) as i32;
        if bd == 8 || x == 255 {
            lut[x]
        } else {
            let start = lut[x] as i32;
            let end = lut[x + 1] as i32;
            (start + round2((end - start) * rem, shift)) as u8
        }
    });
    for (o, v) in out.iter_mut().zip(values) {
        *o = v;
    }
}

/// Spec `Gaussian_Sequence`.
#[rustfmt::skip]
static GAUSSIAN_SEQUENCE: [i16; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224,
    1248, 996, 272, -8, -916, -388, -732, -104, -188, 800, 112, -652,
    -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500, -228,
    12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192,
    1000, -332, 652, -136, -552, -604, -4, 192, -220, -136, 1000, -52,
    372, -96, -624, 124, -24, 396, 540, -12, -104, 640, 464, 244,
    -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292,
    -40, -156, 252, -292, 248, 224, -280, 400, -244, 244, -60, 76,
    -80, 212, 532, 340, 128, -36, 824, -352, -60, -264, -96, -612,
    416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336,
    -96, -792, 304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244,
    60, -948, 0, -708, 268, 108, 356, -548, 488, -344, -136, 488,
    -196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168,
    -108, 464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56,
    120, -408, -116, 436, 504, -232, 328, 844, -164, -84, 784, -168,
    232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296,
    224, 140, -776, -100, 280, 4, 196, 44, -36, -648, 932, 16,
    1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384,
    -448, 208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540,
    28, 136, -88, -436, 256, 296, -1000, 1400, 0, -48, 1056, -136,
    264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768, 388,
    1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204,
    -508, 648, -136, 372, -272, -120, -1004, -552, -548, -384, 548, -296,
    428, -108, -8, -912, -324, -224, -88, -112, -220, -100, 996, -796,
    548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216,
    -412, -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696,
    116, -88, 628, 76, 44, -516, 240, -208, -40, 100, -592, 344,
    -308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
    340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616,
    -752, -100, 560, -1020, 180, -800, -64, 76, 576, 1068, 396, 660,
    552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16, 560,
    516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88,
    -152, 1012, 1064, -228, 164, -376, -684, 592, -392, 156, 196, -524,
    -64, -884, 160, -176, 636, 648, 404, -396, -436, 864, 424, -728,
    988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176,
    -884, 416, -776, -824, -884, 524, -548, -564, -68, -164, -96, 692,
    364, -692, -1012, -68, 260, -480, 876, -1116, 452, -332, -352, 892,
    -1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112,
    -440, -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144,
    -436, 420, 356, 364, -528, 76, 172, -744, -368, 404, -752, -416,
    684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
    -68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12,
    4, -88, 532, 272, -524, 320, 276, -508, 940, 24, -400, -120,
    756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
    328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892,
    -888, 60, 164, 68, -760, 216, -296, 904, -336, -28, 404, -356,
    -568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776, 1156, -428,
    164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72,
    116, 176, -64, -272, 460, -536, -784, -280, 348, 108, -752, -132,
    524, -540, -776, 116, -296, -1196, -288, -560, 1040, -472, 116, -848,
    -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356,
    972, -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828,
    524, 212, -212, 52, 12, 200, 268, -488, -404, -880, 824, -672,
    -40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
    124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268,
    296, 1032, -1196, 436, 316, 372, -432, -200, -660, 704, -224, 596,
    -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4, -1168,
    368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500,
    724, -160, 244, -84, 272, -256, -420, 320, 208, -144, -156, 156,
    364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360, 32,
    -388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840,
    784, -460, 656, -544, -388, -264, 908, -800, -628, -612, -568, 572,
    -220, 164, 288, -16, -308, 308, -112, -636, -760, 280, -668, 432,
    364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580,
    -132, 636, -76, 392, 4, -412, 540, 508, 328, -356, -36, 16,
    -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24, -1044, -32,
    40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356,
    212, -80, -424, -1008, -324, 588, -1496, 576, 460, -816, -848, 56,
    -580, -92, -1372, -112, -496, 200, 364, 52, -140, 48, -48, -60,
    84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164,
    -304, -540, 120, 328, -76, -460, 756, 388, 588, 236, -436, -72,
    -176, -404, -316, -148, 716, -604, 404, -72, -88, -888, -68, 944,
    88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692,
    -508, 132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428,
    552, -692, 60, 356, 96, -384, -188, -612, -576, 736, 508, 892,
    352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508, -144,
    -8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472,
    -204, 908, -188, -1000, -256, 92, 1164, -392, 564, 356, 652, -28,
    -884, 256, 484, -192, 760, -176, 376, -524, -452, -436, 860, -736,
    212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740,
    -240, 400, 132, 20, 192, -196, 264, -668, -1012, -60, 296, -316,
    -828, 76, -156, 284, -768, -448, -832, 148, 248, 652, 616, 1236,
    288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
    -272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260,
    320, 1032, 216, 320, -8, -64, 156, -1016, 1084, 1172, 536, 484,
    -432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
    412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48,
    332, 748, 524, -268, -720, 540, -276, 564, -344, -208, -196, 436,
    896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
    8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876,
    352, -600, -260, -812, -468, -28, -120, -32, -44, 1284, 496, 192,
    464, 312, -76, -516, -380, -456, -1012, -48, 308, -156, 36, 492,
    -156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808,
    -416, 592, 316, -480, 56, 528, -204, -568, 372, -232, 752, -344,
    744, -4, 324, -416, -600, 768, 268, -248, -88, -132, -420, -432,
    80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480,
    -216, -92, 1688, -300, 180, 1020, -176, 820, -68, -228, -260, 436,
    -904, 20, 40, -508, 440, -736, 312, 332, 204, 760, -372, 728,
    96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192,
    396, -728, -520, 276, -188, 80, -52, -612, -252, -48, 648, 212,
    -688, 228, -52, -260, 428, -412, -272, -404, 180, 816, -796, 48,
    152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316,
    476, 12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576,
    260, -352, 120, -16, -136, -260, -492, 72, 556, 660, 580, 616,
    772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
    724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68,
    252, -932, -120, -480, -628, -84, 192, 852, -404, -288, -132, 204,
    100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
    -888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588,
    -184, 228, 12, 372, 48, -848, -344, 224, 208, -200, 484, 128,
    -20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112, -120,
    644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84,
    200, -152, 0, -144, 584, 280, -120, 80, -556, -972, -196, -472,
    724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372, -776,
    740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956,
    236, 356, 20, -40, 300, 24, -676, -596, 132, 1120, -104, 532,
    -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488, 424, 24,
    756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284,
    1144, 1160, 600, 424, 888, 656, -356, -320, 220, 316, -176, -724,
    -188, -816, -628, -348, -228, -380, 1012, -452, -660, 736, 928, 404,
    -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344,
    428, 548, -112, 136, -228, -216, -820, -516, 340, 92, -136, 116,
    -300, 376, -244, 100, -316, -520, -284, -12, 824, 164, -548, -180,
    -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
    1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548,
    -148, 504, 316, 152, -648, -620, -684, -24, -376, -384, -108, -920,
    -1032, 768, 180, -264, -508, -1268, -260, -60, 300, -240, 988, 724,
    -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4,
    -216, -32, 836, 268, 396, 1332, 864, -600, 100, 56, -412, -92,
    356, 180, 884, -468, -436, 292, -388, -804, -704, -840, 368, -348,
    140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32,
    -228, 132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244,
    12, -736, -296, 360, 468, -376, -108, -92, 788, 368, -56, 544,
    400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
    204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200,
    -56, 276, -88, 484, -324, 948, 864, 1000, -456, -184, -276, 292,
    -296, 156, 676, 320, 160, 908, -84, -1236, -288, -116, 260, -372,
    -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216,
    -1044, -136, -676, -396, -1500, 960, -40, 176, 168, 1516, 420, -504,
    -344, -364, -360, 1216, -940, -380, -212, 252, -660, -708, 484, -444,
    -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
    -288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196,
    76, 388, 232, 412, -1168, -716, -644, 756, -172, -356, -504, 116,
    432, 528, 48, 476, -168, -608, 448, 160, -532, -272, 28, -676,
    -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368,
    -52, -524, -572, -556, -200, 768, 1124, -208, -512, 176, 232, 248,
    -148, -888, 604, -600, -304, 804, -156, -212, 488, -192, -804, -256,
    368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572,
    -12, -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848,
    -592, 108, -76, 244, 312, -716, 592, -80, 436, 360, 4, -248,
    160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
    912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624,
    300, -212, -1144, 32, -724, 800, -1128, -212, -1288, -848, 180, -416,
    440, 192, -576, -792, -76, -1080, 80, -532, -352, -132, 380, -820,
    148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832,
    508, 552, -52, -100, -656, 208, -568, 748, -88, 680, 232, 300,
    192, -408, -1012, -152, -252, -268, 272, -876, -664, -648, -332, -136,
    16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
    228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92,
    356, 404, -408, 252, 196, -176, -556, 792, 268, 32, 372, 40,
    96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
    656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32,
    252, 0, -412, 188, -696, 508, -476, 324, -1096, 656, -312, 560,
    264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348, -288,
    -276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52,
    860, 884, 212, -12, 168, 1020, 512, -552, 924, -148, 716, 188,
    164, -340, -520, -184, 880, -152, -680, -208, -1156, -300, -528, -472,
    364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280,
    -224, 96, 568, -76, 172, 148, 148, 104, 32, -296, -32, 788,
    -80, 32, -16, 280, 288, 944, 428, -484,
];

#[cfg(test)]
// The spec transcription below is intentionally literal.
#[allow(
    clippy::identity_op,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::collapsible_else_if
)]
mod tests {
    use super::*;

    /// Small deterministic PRNG for tests.
    struct TestRng(u64);
    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn bool(&mut self) -> bool {
            self.next() & 1 == 1
        }
    }

    /// Random strictly increasing points.
    fn random_points(r: &mut TestRng, n: usize, values: &mut [u8; 16], scaling: &mut [u8; 16]) {
        // Pick n distinct values in 0..=255, sorted.
        let mut chosen = [false; 256];
        let mut count = 0;
        while count < n {
            let v = r.below(256) as usize;
            if !chosen[v] {
                chosen[v] = true;
                count += 1;
            }
        }
        let mut i = 0;
        for (v, &c) in chosen.iter().enumerate() {
            if c {
                values[i] = v as u8;
                scaling[i] = r.below(256) as u8;
                i += 1;
            }
        }
    }

    fn random_params(r: &mut TestRng, mono: bool, sub_x: bool, sub_y: bool) -> FilmGrainParams {
        let mut p = FilmGrainParams {
            apply_grain: true,
            grain_seed: r.next() as u16,
            ..Default::default()
        };
        p.num_y_points = if r.below(5) == 0 { 0 } else { r.below(15) as usize };
        random_points(
            r,
            p.num_y_points,
            &mut p.point_y_value,
            &mut p.point_y_scaling,
        );
        p.chroma_scaling_from_luma = !mono && r.below(4) == 0;
        if !(mono || p.chroma_scaling_from_luma || (sub_x && sub_y && p.num_y_points == 0)) {
            p.num_cb_points = if r.below(4) == 0 { 0 } else { r.below(11) as usize };
            p.num_cr_points = if r.below(4) == 0 { 0 } else { r.below(11) as usize };
            if sub_x && sub_y {
                // 4:2:0 requires both or neither.
                if (p.num_cb_points == 0) != (p.num_cr_points == 0) {
                    p.num_cr_points = p.num_cb_points;
                }
            }
            random_points(
                r,
                p.num_cb_points,
                &mut p.point_cb_value,
                &mut p.point_cb_scaling,
            );
            random_points(
                r,
                p.num_cr_points,
                &mut p.point_cr_value,
                &mut p.point_cr_scaling,
            );
        }
        p.grain_scaling_minus_8 = r.below(4) as u8;
        p.ar_coeff_lag = r.below(4) as u8;
        // Keep AR coefficients mostly small (realistic) but sometimes full
        // range to exercise clipping.
        let wide = r.below(3) == 0;
        let coef = |r: &mut TestRng| {
            if wide {
                r.below(256) as u8
            } else {
                (128 + r.below(41) as i32 - 20) as u8
            }
        };
        for c in p.ar_coeffs_y_plus_128.iter_mut() {
            *c = coef(r);
        }
        for c in p.ar_coeffs_cb_plus_128.iter_mut() {
            *c = coef(r);
        }
        for c in p.ar_coeffs_cr_plus_128.iter_mut() {
            *c = coef(r);
        }
        p.ar_coeff_shift_minus_6 = r.below(4) as u8;
        p.grain_scale_shift = r.below(4) as u8;
        p.cb_mult = r.below(256) as u8;
        p.cb_luma_mult = r.below(256) as u8;
        p.cb_offset = r.below(512) as u16;
        p.cr_mult = r.below(256) as u8;
        p.cr_luma_mult = r.below(256) as u8;
        p.cr_offset = r.below(512) as u16;
        p.overlap_flag = r.bool();
        p.clip_to_restricted_range = r.bool();
        p
    }

    // ---------------------------------------------------------------
    // Direct transcription of spec 7.18.3 (whole frame at once).
    // ---------------------------------------------------------------

    struct Spec<'a> {
        p: &'a FilmGrainParams,
        bd: i32,
        mono: bool,
        sub_x: usize,
        sub_y: usize,
        mc_identity: bool,
        random_register: u32,
        grain_min: i32,
        grain_max: i32,
        luma_grain: Vec<Vec<i32>>,
        cb_grain: Vec<Vec<i32>>,
        cr_grain: Vec<Vec<i32>>,
        scaling_lut: [[i32; 256]; 3],
    }

    fn sround2(x: i32, n: i32) -> i32 {
        if n == 0 {
            x
        } else {
            (x + (1 << (n - 1))) >> n
        }
    }

    fn sclip3(lo: i32, hi: i32, x: i32) -> i32 {
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    }

    impl<'a> Spec<'a> {
        fn get_random_number(&mut self, bits: u32) -> i32 {
            let mut r = self.random_register;
            let bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
            r = (r >> 1) | (bit << 15);
            let result = (r >> (16 - bits)) & ((1 << bits) - 1);
            self.random_register = r;
            result as i32
        }

        fn generate_grain(&mut self) {
            let p = self.p;
            let shift = 12 - self.bd + p.grain_scale_shift as i32;
            self.luma_grain = vec![vec![0; 82]; 73];
            for y in 0..73 {
                for x in 0..82 {
                    let g = if p.num_y_points > 0 {
                        GAUSSIAN_SEQUENCE[self.get_random_number(11) as usize] as i32
                    } else {
                        0
                    };
                    self.luma_grain[y][x] = sround2(g, shift);
                }
            }
            let shift = p.ar_coeff_shift_minus_6 as i32 + 6;
            let lag = p.ar_coeff_lag as i32;
            for y in 3..73i32 {
                for x in 3..(82 - 3) as i32 {
                    let mut s = 0;
                    let mut pos = 0;
                    'a: for delta_row in -lag..=0 {
                        for delta_col in -lag..=lag {
                            if delta_row == 0 && delta_col == 0 {
                                break 'a;
                            }
                            let c = p.ar_coeffs_y_plus_128[pos] as i32 - 128;
                            s += self.luma_grain[(y + delta_row) as usize]
                                [(x + delta_col) as usize]
                                * c;
                            pos += 1;
                        }
                    }
                    let v = self.luma_grain[y as usize][x as usize] + sround2(s, shift);
                    self.luma_grain[y as usize][x as usize] =
                        sclip3(self.grain_min, self.grain_max, v);
                }
            }
            if self.mono {
                return;
            }
            let chroma_w = if self.sub_x == 1 { 44 } else { 82 };
            let chroma_h = if self.sub_y == 1 { 38 } else { 73 };
            let shift = 12 - self.bd + p.grain_scale_shift as i32;
            self.cb_grain = vec![vec![0; chroma_w]; chroma_h];
            self.cr_grain = vec![vec![0; chroma_w]; chroma_h];
            self.random_register = (p.grain_seed ^ 0xb524) as u32;
            for y in 0..chroma_h {
                for x in 0..chroma_w {
                    let g = if p.num_cb_points > 0 || p.chroma_scaling_from_luma {
                        GAUSSIAN_SEQUENCE[self.get_random_number(11) as usize] as i32
                    } else {
                        0
                    };
                    self.cb_grain[y][x] = sround2(g, shift);
                }
            }
            self.random_register = (p.grain_seed ^ 0x49d8) as u32;
            for y in 0..chroma_h {
                for x in 0..chroma_w {
                    let g = if p.num_cr_points > 0 || p.chroma_scaling_from_luma {
                        GAUSSIAN_SEQUENCE[self.get_random_number(11) as usize] as i32
                    } else {
                        0
                    };
                    self.cr_grain[y][x] = sround2(g, shift);
                }
            }
            let shift = p.ar_coeff_shift_minus_6 as i32 + 6;
            for y in 3..chroma_h as i32 {
                for x in 3..chroma_w as i32 - 3 {
                    let mut s0 = 0;
                    let mut s1 = 0;
                    let mut pos = 0;
                    'b: for delta_row in -lag..=0 {
                        for delta_col in -lag..=lag {
                            let c0 = p.ar_coeffs_cb_plus_128[pos] as i32 - 128;
                            let c1 = p.ar_coeffs_cr_plus_128[pos] as i32 - 128;
                            if delta_row == 0 && delta_col == 0 {
                                if p.num_y_points > 0 {
                                    let mut luma = 0;
                                    let luma_x = ((x - 3) << self.sub_x) + 3;
                                    let luma_y = ((y - 3) << self.sub_y) + 3;
                                    for i in 0..=self.sub_y as i32 {
                                        for j in 0..=self.sub_x as i32 {
                                            luma += self.luma_grain[(luma_y + i) as usize]
                                                [(luma_x + j) as usize];
                                        }
                                    }
                                    luma = sround2(luma, (self.sub_x + self.sub_y) as i32);
                                    s0 += luma * c0;
                                    s1 += luma * c1;
                                }
                                break 'b;
                            }
                            let yy = (y + delta_row) as usize;
                            let xx = (x + delta_col) as usize;
                            s0 += self.cb_grain[yy][xx] * c0;
                            s1 += self.cr_grain[yy][xx] * c1;
                            pos += 1;
                        }
                    }
                    let (yu, xu) = (y as usize, x as usize);
                    self.cb_grain[yu][xu] = sclip3(
                        self.grain_min,
                        self.grain_max,
                        self.cb_grain[yu][xu] + sround2(s0, shift),
                    );
                    self.cr_grain[yu][xu] = sclip3(
                        self.grain_min,
                        self.grain_max,
                        self.cr_grain[yu][xu] + sround2(s1, shift),
                    );
                }
            }
        }

        fn get_x(&self, plane: usize, i: usize) -> i32 {
            let p = self.p;
            (if plane == 0 || p.chroma_scaling_from_luma {
                p.point_y_value[i]
            } else if plane == 1 {
                p.point_cb_value[i]
            } else {
                p.point_cr_value[i]
            }) as i32
        }

        fn get_y(&self, plane: usize, i: usize) -> i32 {
            let p = self.p;
            (if plane == 0 || p.chroma_scaling_from_luma {
                p.point_y_scaling[i]
            } else if plane == 1 {
                p.point_cb_scaling[i]
            } else {
                p.point_cr_scaling[i]
            }) as i32
        }

        fn init_scaling(&mut self) {
            let p = self.p;
            let num_planes = if self.mono { 1 } else { 3 };
            for plane in 0..num_planes {
                let num_points = if plane == 0 || p.chroma_scaling_from_luma {
                    p.num_y_points
                } else if plane == 1 {
                    p.num_cb_points
                } else {
                    p.num_cr_points
                };
                if num_points == 0 {
                    for x in 0..256 {
                        self.scaling_lut[plane][x] = 0;
                    }
                } else {
                    for x in 0..self.get_x(plane, 0) {
                        self.scaling_lut[plane][x as usize] = self.get_y(plane, 0);
                    }
                    for i in 0..num_points - 1 {
                        let delta_y = self.get_y(plane, i + 1) - self.get_y(plane, i);
                        let delta_x = self.get_x(plane, i + 1) - self.get_x(plane, i);
                        let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
                        for x in 0..delta_x {
                            let v = self.get_y(plane, i) + ((x * delta + 32768) >> 16);
                            self.scaling_lut[plane][(self.get_x(plane, i) + x) as usize] = v;
                        }
                    }
                    for x in self.get_x(plane, num_points - 1)..256 {
                        self.scaling_lut[plane][x as usize] = self.get_y(plane, num_points - 1);
                    }
                }
            }
        }

        fn scale_lut(&self, plane: usize, index: i32) -> i32 {
            let shift = self.bd - 8;
            let x = index >> shift;
            let rem = index - (x << shift);
            if self.bd == 8 || x == 255 {
                self.scaling_lut[plane][x as usize]
            } else {
                let start = self.scaling_lut[plane][x as usize];
                let end = self.scaling_lut[plane][x as usize + 1];
                start + sround2((end - start) * rem, shift)
            }
        }

        /// 7.18.3.5 on full planes `out[plane][y][x]`.
        fn add_noise(&mut self, out: &mut [Vec<Vec<i32>>], w: usize, h: usize) {
            let p = self.p;
            let num_planes = if self.mono { 1 } else { 3 };
            let (sub_x, sub_y) = (self.sub_x, self.sub_y);
            let num_stripes = (h + 1) / 2 / 16 + 2;
            // noise_stripe[lumaNum][plane][i][x], generously sized.
            let mut noise_stripe = vec![vec![vec![vec![0i32; w + 64]; 34]; 3]; num_stripes];
            let mut luma_num = 0;
            let mut y = 0;
            while y < (h + 1) / 2 {
                self.random_register = p.grain_seed as u32;
                self.random_register ^= (((luma_num * 37 + 178) & 255) << 8) as u32;
                self.random_register ^= ((luma_num * 173 + 105) & 255) as u32;
                let mut x = 0;
                while x < (w + 1) / 2 {
                    let rand = self.get_random_number(8) as usize;
                    let offset_x = rand >> 4;
                    let offset_y = rand & 15;
                    for plane in 0..num_planes {
                        let plane_sub_x = if plane > 0 { sub_x } else { 0 };
                        let plane_sub_y = if plane > 0 { sub_y } else { 0 };
                        let plane_offset_x =
                            if plane_sub_x != 0 { 6 + offset_x } else { 9 + offset_x * 2 };
                        let plane_offset_y =
                            if plane_sub_y != 0 { 6 + offset_y } else { 9 + offset_y * 2 };
                        for i in 0..(34 >> plane_sub_y) {
                            for j in 0..(34 >> plane_sub_x) {
                                let mut g = if plane == 0 {
                                    self.luma_grain[plane_offset_y + i][plane_offset_x + j]
                                } else if plane == 1 {
                                    self.cb_grain[plane_offset_y + i][plane_offset_x + j]
                                } else {
                                    self.cr_grain[plane_offset_y + i][plane_offset_x + j]
                                };
                                if plane_sub_x == 0 {
                                    if j < 2 && p.overlap_flag && x > 0 {
                                        let old = noise_stripe[luma_num][plane][i][x * 2 + j];
                                        if j == 0 {
                                            g = old * 27 + g * 17;
                                        } else {
                                            g = old * 17 + g * 27;
                                        }
                                        g = sclip3(self.grain_min, self.grain_max, sround2(g, 5));
                                    }
                                    noise_stripe[luma_num][plane][i][x * 2 + j] = g;
                                } else {
                                    if j == 0 && p.overlap_flag && x > 0 {
                                        let old = noise_stripe[luma_num][plane][i][x + j];
                                        g = old * 23 + g * 22;
                                        g = sclip3(self.grain_min, self.grain_max, sround2(g, 5));
                                    }
                                    noise_stripe[luma_num][plane][i][x + j] = g;
                                }
                            }
                        }
                    }
                    x += 16;
                }
                luma_num += 1;
                y += 16;
            }

            let mut noise_image = vec![vec![vec![0i32; w]; h]; 3];
            for plane in 0..num_planes {
                let plane_sub_x = if plane > 0 { sub_x } else { 0 };
                let plane_sub_y = if plane > 0 { sub_y } else { 0 };
                for y in 0..((h + plane_sub_y) >> plane_sub_y) {
                    let luma_num = y >> (5 - plane_sub_y);
                    let i = y - (luma_num << (5 - plane_sub_y));
                    for x in 0..((w + plane_sub_x) >> plane_sub_x) {
                        let mut g = noise_stripe[luma_num][plane][i][x];
                        if plane_sub_y == 0 {
                            if i < 2 && luma_num > 0 && p.overlap_flag {
                                let old = noise_stripe[luma_num - 1][plane][i + 32][x];
                                if i == 0 {
                                    g = old * 27 + g * 17;
                                } else {
                                    g = old * 17 + g * 27;
                                }
                                g = sclip3(self.grain_min, self.grain_max, sround2(g, 5));
                            }
                        } else if i < 1 && luma_num > 0 && p.overlap_flag {
                            let old = noise_stripe[luma_num - 1][plane][i + 16][x];
                            g = old * 23 + g * 22;
                            g = sclip3(self.grain_min, self.grain_max, sround2(g, 5));
                        }
                        noise_image[plane][y][x] = g;
                    }
                }
            }

            let bd = self.bd;
            let (min_value, max_luma, max_chroma);
            if p.clip_to_restricted_range {
                min_value = 16 << (bd - 8);
                max_luma = 235 << (bd - 8);
                max_chroma = if self.mc_identity { max_luma } else { 240 << (bd - 8) };
            } else {
                min_value = 0;
                max_luma = (256 << (bd - 8)) - 1;
                max_chroma = max_luma;
            }
            let clip1 = |x: i32| sclip3(0, (1 << bd) - 1, x);
            let scaling_shift = p.grain_scaling_minus_8 as i32 + 8;
            if num_planes > 1 {
                for y in 0..((h + sub_y) >> sub_y) {
                    for x in 0..((w + sub_x) >> sub_x) {
                        let luma_x = x << sub_x;
                        let luma_y = y << sub_y;
                        let luma_next_x = (luma_x + 1).min(w - 1);
                        let average_luma = if sub_x != 0 {
                            sround2(out[0][luma_y][luma_x] + out[0][luma_y][luma_next_x], 1)
                        } else {
                            out[0][luma_y][luma_x]
                        };
                        if p.num_cb_points > 0 || p.chroma_scaling_from_luma {
                            let orig = out[1][y][x];
                            let merged = if p.chroma_scaling_from_luma {
                                average_luma
                            } else {
                                let combined = average_luma * (p.cb_luma_mult as i32 - 128)
                                    + orig * (p.cb_mult as i32 - 128);
                                clip1((combined >> 6) + ((p.cb_offset as i32 - 256) << (bd - 8)))
                            };
                            let noise = noise_image[1][y][x];
                            let noise = sround2(self.scale_lut(1, merged) * noise, scaling_shift);
                            out[1][y][x] = sclip3(min_value, max_chroma, orig + noise);
                        }
                        if p.num_cr_points > 0 || p.chroma_scaling_from_luma {
                            let orig = out[2][y][x];
                            let merged = if p.chroma_scaling_from_luma {
                                average_luma
                            } else {
                                let combined = average_luma * (p.cr_luma_mult as i32 - 128)
                                    + orig * (p.cr_mult as i32 - 128);
                                clip1((combined >> 6) + ((p.cr_offset as i32 - 256) << (bd - 8)))
                            };
                            let noise = noise_image[2][y][x];
                            let noise = sround2(self.scale_lut(2, merged) * noise, scaling_shift);
                            out[2][y][x] = sclip3(min_value, max_chroma, orig + noise);
                        }
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let orig = out[0][y][x];
                    let noise = noise_image[0][y][x];
                    let noise = sround2(self.scale_lut(0, orig) * noise, scaling_shift);
                    if p.num_y_points > 0 {
                        out[0][y][x] = sclip3(min_value, max_luma, orig + noise);
                    }
                }
            }
        }
    }

    /// Spec 7.18.3 applied to whole planes.
    #[allow(clippy::too_many_arguments)]
    fn spec_film_grain(
        p: &FilmGrainParams,
        bd: u32,
        mono: bool,
        sub_x: bool,
        sub_y: bool,
        mc_identity: bool,
        out: &mut [Vec<Vec<i32>>],
        w: usize,
        h: usize,
    ) {
        let grain_center = 128 << (bd - 8);
        let mut s = Spec {
            p,
            bd: bd as i32,
            mono,
            sub_x: sub_x as usize,
            sub_y: sub_y as usize,
            mc_identity,
            random_register: p.grain_seed as u32,
            grain_min: -grain_center,
            grain_max: (256 << (bd - 8)) - 1 - grain_center,
            luma_grain: Vec::new(),
            cb_grain: Vec::new(),
            cr_grain: Vec::new(),
            scaling_lut: [[0; 256]; 3],
        };
        s.generate_grain();
        s.init_scaling();
        s.add_noise(out, w, h);
    }

    /// Random frame planes (`[plane][y][x]`).
    fn random_frame(
        r: &mut TestRng,
        bd: u32,
        mono: bool,
        sx: usize,
        sy: usize,
        w: usize,
        h: usize,
    ) -> Vec<Vec<Vec<i32>>> {
        let num_planes = if mono { 1 } else { 3 };
        let max = 1u64 << bd;
        // Smooth-ish content with random noise, sometimes extremes.
        (0..num_planes)
            .map(|pl| {
                let (pw, ph) = if pl == 0 { (w, h) } else { ((w + sx) >> sx, (h + sy) >> sy) };
                let mode = r.below(3);
                (0..ph)
                    .map(|_| {
                        (0..pw)
                            .map(|_| match mode {
                                0 => r.below(max) as i32,
                                1 => [0, max as i32 - 1][r.below(2) as usize],
                                _ => (max as i32 / 2 + r.below(64) as i32 - 32)
                                    .clamp(0, max as i32 - 1),
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// Runs `FilmGrain::apply_band` on `frame` with the given band height
    /// (`None`: whole frame at once) and returns the result.
    #[allow(clippy::too_many_arguments)]
    fn run_impl<T: Pixel>(
        fg: &FilmGrain,
        frame: &[Vec<Vec<i32>>],
        sx: usize,
        sy: usize,
        w: usize,
        h: usize,
        band: Option<usize>,
        pad: usize,
    ) -> Vec<Vec<Vec<i32>>> {
        let num_planes = frame.len();
        let dims: Vec<(usize, usize)> = (0..num_planes)
            .map(
                |p| {
                    if p == 0 {
                        (w, h)
                    } else {
                        ((w + sx) >> sx, (h + sy) >> sy)
                    }
                },
            )
            .collect();
        let strides: Vec<usize> = dims.iter().map(|d| d.0 + pad).collect();
        let mut bufs: Vec<Vec<T>> = (0..num_planes)
            .map(|p| {
                let mut b = vec![T::default(); strides[p] * dims[p].1];
                for y in 0..dims[p].1 {
                    for x in 0..dims[p].0 {
                        b[y * strides[p] + x] = T::from_i32(frame[p][y][x]);
                    }
                }
                b
            })
            .collect();
        let band_h = band.unwrap_or(h);
        let mut y0 = 0;
        while y0 < h {
            let bh = band_h.min(h - y0);
            let mut slices: Vec<&mut [T]> = bufs
                .iter_mut()
                .enumerate()
                .map(|(p, b)| {
                    let (ssy, ry0, ry1) = if p == 0 {
                        (0, y0, y0 + bh)
                    } else {
                        (sy, y0 >> sy, (y0 + bh + sy) >> sy)
                    };
                    let _ = ssy;
                    // Give each band exactly its rows (the last row may be
                    // short of padding).
                    let start = ry0 * strides[p];
                    let end = ((ry1 - 1) * strides[p] + dims[p].0).min(b.len());
                    &mut b[start..end]
                })
                .collect();
            fg.apply_band(&mut slices, &strides, y0, bh, w, h);
            y0 += bh;
        }
        (0..num_planes)
            .map(|p| {
                (0..dims[p].1)
                    .map(|y| {
                        (0..dims[p].0)
                            .map(|x| bufs[p][y * strides[p] + x].to_i32())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    const SUBSAMPLINGS: [(bool, bool, bool); 4] = [
        (true, false, false),
        (false, true, true),
        (false, true, false),
        (false, false, false),
    ];

    fn check(seed: u64, iters: usize, sizes: &[(usize, usize)]) {
        let mut r = TestRng(seed);
        let (mut total, mut changed) = (0, 0);
        for it in 0..iters {
            for &(mono, sub_x, sub_y) in SUBSAMPLINGS.iter() {
                for &bd in &[8u32, 10, 12] {
                    let (w, h) = sizes[(it + bd as usize) % sizes.len()];
                    let p = random_params(&mut r, mono, sub_x, sub_y);
                    let mc_identity = r.bool();
                    let (sx, sy) = (sub_x as usize, sub_y as usize);
                    let frame = random_frame(&mut r, bd, mono, sx, sy, w, h);
                    let mut expected = frame.clone();
                    spec_film_grain(&p, bd, mono, sub_x, sub_y, mc_identity, &mut expected, w, h);
                    total += 1;
                    changed += (expected != frame) as usize;
                    let fg = FilmGrain::new(&p, bd, mono, sub_x, sub_y, mc_identity);
                    let bands = [None, Some(32), Some(64), Some(96)];
                    for band in bands {
                        let pad = r.below(3) as usize * 7;
                        let got = if bd == 8 {
                            run_impl::<u8>(&fg, &frame, sx, sy, w, h, band, pad)
                        } else {
                            run_impl::<u16>(&fg, &frame, sx, sy, w, h, band, pad)
                        };
                        assert!(
                            got == expected,
                            "mismatch: seed {seed} it {it} bd {bd} mono {mono} sub {sub_x},{sub_y} size {w}x{h} band {band:?} params {p:?}"
                        );
                    }
                }
            }
        }
        // Make sure the test is meaningful: most configurations add noise.
        assert!(
            changed * 2 > total,
            "only {changed} of {total} configurations changed the frame"
        );
    }

    #[test]
    fn rng_matches_spec() {
        let mut r = Rng(0x1234);
        let mut reg: u32 = 0x1234;
        for bits in [8u32, 11, 1, 16, 3].iter().cycle().take(1000) {
            let bit = ((reg >> 0) ^ (reg >> 1) ^ (reg >> 3) ^ (reg >> 12)) & 1;
            reg = (reg >> 1) | (bit << 15);
            let expected = ((reg >> (16 - bits)) & ((1 << bits) - 1)) as i32;
            assert_eq!(r.next(*bits), expected);
        }
    }

    #[test]
    fn spec_equivalence_small_and_odd_sizes() {
        let sizes = [
            (1, 1),
            (2, 3),
            (7, 5),
            (31, 33),
            (33, 31),
            (32, 32),
            (17, 65),
            (65, 17),
            (34, 66),
        ];
        check(0x9e3779b97f4a7c15, 40, &sizes);
    }

    #[test]
    fn spec_equivalence_larger_sizes() {
        let sizes = [
            (97, 129),
            (128, 96),
            (161, 101),
            (100, 163),
            (255, 66),
            (521, 41),
        ];
        check(0x1234_5678_9abc_def1, 10, &sizes);
    }

    #[test]
    fn no_grain_is_identity() {
        let p = FilmGrainParams {
            apply_grain: true,
            grain_seed: 77,
            ..Default::default()
        };
        let fg = FilmGrain::new(&p, 10, false, true, true, false);
        let mut r = TestRng(5);
        let frame = random_frame(&mut r, 10, false, 1, 1, 45, 37);
        let got = run_impl::<u16>(&fg, &frame, 1, 1, 45, 37, Some(32), 3);
        assert_eq!(got, frame);
    }
}
