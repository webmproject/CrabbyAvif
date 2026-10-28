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

//! Output sink: optional film grain, crop and downscale of the finished
//! rows produced by the decoding pipeline.

use crate::decoder::av1::decode::{PlaneRows, RowSink};
use crate::decoder::av1::dsp::film_grain::FilmGrain;
use crate::decoder::av1::error::Result;
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::scale::PlaneScaler;

/// A plane of output samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane<T> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    /// Index of sample (0, 0) in `data` (decoder planes start at a 64-byte
    /// aligned address inside their allocation).
    pub offset: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Plane<T> {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            stride: width,
            offset: 0,
            data: vec![T::default(); width * height],
        }
    }

    /// The samples from (0, 0) on, rows `stride` apart.
    pub fn samples(&self) -> &[T] {
        &self.data[self.offset..]
    }

    pub fn samples_mut(&mut self) -> &mut [T] {
        &mut self.data[self.offset..]
    }

    pub fn row(&self, y: usize) -> &[T] {
        let i = self.offset + y * self.stride;
        &self.data[i..i + self.width]
    }

    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        let i = self.offset + y * self.stride;
        &mut self.data[i..i + self.width]
    }
}

/// Crop rectangle of one plane (in plane samples).
#[derive(Clone, Copy, Debug)]
pub struct PlaneCrop {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

pub struct OutputSink<T: Pixel> {
    crops: Vec<PlaneCrop>,
    scalers: Vec<Option<PlaneScaler>>,
    pub planes: Vec<Plane<T>>,
    grain: Option<GrainStage<T>>,
    /// The output planes are the whole frame (no crop or scaling), so
    /// grain is applied in place there instead of in `GrainStage::buf`.
    grain_in_place: bool,
    /// The output is the whole frame (no crop or scaling).
    full: bool,
    /// Next row expected per plane.
    next_y: Vec<usize>,
}

struct GrainStage<T> {
    fg: FilmGrain,
    frame_w: usize,
    frame_h: usize,
    sub_y: usize,
    /// Buffered full-width rows per plane, starting at luma row `y0`.
    buf: Vec<Vec<T>>,
    widths: Vec<usize>,
    y0: usize,
    rows: usize,
}

impl<T: Pixel> OutputSink<T> {
    /// `plane_dims`: full frame plane sizes; `crops`: crop per plane;
    /// `out_dims`: output size per plane (equal to the crop size if no
    /// scaling).
    pub fn new(
        plane_dims: &[(usize, usize)],
        crops: Vec<PlaneCrop>,
        out_dims: &[(usize, usize)],
        grain: Option<(FilmGrain, usize)>,
        bd: u32,
    ) -> Self {
        let _ = bd;
        let mut scalers = Vec::new();
        for (p, c) in crops.iter().enumerate() {
            let (ow, oh) = out_dims[p];
            if ow != c.w || oh != c.h {
                scalers.push(Some(PlaneScaler::new(c.w, c.h, ow, oh)));
            } else {
                scalers.push(None);
            }
        }
        let full = crops
            .iter()
            .zip(plane_dims)
            .zip(&scalers)
            .all(|((c, &(w, h)), s)| s.is_none() && c.x0 == 0 && c.y0 == 0 && c.w == w && c.h == h);
        // When `full` is set, `decode_frame` hands its planes to
        // `finish` instead of calling `push` (unless superres is active,
        // in which case `forward` allocates on first use).
        let planes = out_dims
            .iter()
            .map(|&(ow, oh)| {
                if full {
                    Plane {
                        width: ow,
                        height: oh,
                        stride: ow,
                        offset: 0,
                        data: Vec::new(),
                    }
                } else {
                    Plane::new(ow, oh)
                }
            })
            .collect();
        let grain_in_place = full && grain.is_some();
        let grain = grain.map(|(fg, sub_y)| GrainStage {
            fg,
            frame_w: plane_dims[0].0,
            frame_h: plane_dims[0].1,
            sub_y,
            buf: if grain_in_place {
                Vec::new()
            } else {
                plane_dims
                    .iter()
                    .map(|&(w, _)| vec![T::default(); w * 96])
                    .collect()
            },
            widths: plane_dims.iter().map(|&(w, _)| w).collect(),
            y0: 0,
            rows: 0,
        });
        Self {
            next_y: vec![0; crops.len()],
            crops,
            scalers,
            planes,
            grain,
            grain_in_place,
            full,
        }
    }

    fn forward(&mut self, bands: &[PlaneRows<'_, T>]) {
        for (p, b) in bands.iter().enumerate() {
            let out = &mut self.planes[p];
            if out.data.is_empty() {
                out.data.resize(out.stride * out.height, T::default());
            }
            self.next_y[p] = b.y0 + b.h;
            let c = self.crops[p];
            for i in 0..b.h {
                let y = b.y0 + i;
                if y < c.y0 || y >= c.y0 + c.h {
                    continue;
                }
                let row = &b.data[i * b.stride + c.x0..i * b.stride + c.x0 + c.w];
                let out = &mut self.planes[p];
                match &mut self.scalers[p] {
                    Some(s) => {
                        let ow = out.width;
                        s.push_row(
                            &row.iter().map(|v| v.to_u16()).collect::<Vec<u16>>(),
                            |oy, r| {
                                let dst = &mut out.data[oy * ow..(oy + 1) * ow];
                                for (d, &v) in dst.iter_mut().zip(r.iter()) {
                                    *d = T::from_i32(v as i32);
                                }
                            },
                        );
                    }
                    None => out.row_mut(y - c.y0).copy_from_slice(row),
                }
            }
        }
    }
}

impl<T: Pixel> RowSink<T> for OutputSink<T> {
    fn wants_frame(&self) -> bool {
        // Full frame output without scaling (`grain_in_place` without
        // grain).
        self.full
    }

    fn finish(&mut self, mut planes: Vec<Plane<T>>) -> Result<()> {
        debug_assert!(self.full);
        if let Some(g) = self.grain.as_mut() {
            // Rows not yet handled by `rows_final` (all of them if the
            // decoder did not call it).
            if g.y0 < g.frame_h {
                let sub_y = g.sub_y;
                let y0 = g.y0;
                let strides: Vec<usize> = planes.iter().map(|p| p.stride).collect();
                let mut refs: Vec<&mut [T]> = planes
                    .iter_mut()
                    .enumerate()
                    .map(|(p, pl)| {
                        let sy = if p > 0 { sub_y } else { 0 };
                        let off = (y0 >> sy) * pl.stride;
                        &mut pl.samples_mut()[off..]
                    })
                    .collect();
                g.fg.apply_band(
                    &mut refs,
                    &strides,
                    y0,
                    g.frame_h - y0,
                    g.frame_w,
                    g.frame_h,
                );
                g.y0 = g.frame_h;
            }
        }
        self.planes = planes;
        Ok(())
    }

    fn rows_final(&mut self, planes: &mut [&mut [T]], strides: &[usize], y_end: usize) {
        if !self.full {
            return;
        }
        let Some(g) = self.grain.as_mut() else {
            return;
        };
        // Film grain works on 32 luma row blocks.
        let y_end = y_end.min(g.frame_h);
        if y_end <= g.y0 {
            return;
        }
        let chunk = if y_end == g.frame_h { y_end - g.y0 } else { (y_end - g.y0) / 32 * 32 };
        if chunk == 0 {
            return;
        }
        let sub_y = g.sub_y;
        let y0 = g.y0;
        let mut refs: Vec<&mut [T]> = planes
            .iter_mut()
            .zip(strides)
            .enumerate()
            .map(|(p, (pl, &stride))| {
                let sy = if p > 0 { sub_y } else { 0 };
                &mut pl[(y0 >> sy) * stride..]
            })
            .collect();
        g.fg.apply_band(&mut refs, strides, y0, chunk, g.frame_w, g.frame_h);
        g.y0 += chunk;
    }

    fn push(&mut self, bands: &[PlaneRows<'_, T>]) -> Result<()> {
        let Some(g) = self.grain.as_mut() else {
            self.forward(bands);
            return Ok(());
        };
        if self.grain_in_place {
            // Output planes are the full frame: copy the rows there and
            // apply the grain in place to every complete 32 row block.
            self.forward(bands);
            let OutputSink { planes, grain, .. } = self;
            let g = grain.as_mut().unwrap();
            g.rows += bands[0].h;
            let end = g.y0 + g.rows == g.frame_h;
            let chunk = if end { g.rows } else { g.rows / 32 * 32 };
            if chunk == 0 {
                return Ok(());
            }
            let sub_y = g.sub_y;
            let y0 = g.y0;
            let strides: Vec<usize> = planes.iter().map(|p| p.stride).collect();
            let mut refs: Vec<&mut [T]> = planes
                .iter_mut()
                .enumerate()
                .map(|(p, pl)| {
                    let sy = if p > 0 { sub_y } else { 0 };
                    let off = (y0 >> sy) * pl.stride;
                    &mut pl.samples_mut()[off..]
                })
                .collect();
            g.fg.apply_band(&mut refs, &strides, y0, chunk, g.frame_w, g.frame_h);
            g.y0 += chunk;
            g.rows -= chunk;
            return Ok(());
        }
        // Append to the grain buffer. Film grain works on 32 luma row
        // blocks: when output starts in the middle of the frame (region
        // decode), rows up to the next multiple of 32 are dropped (they are
        // above the requested region).
        let mut skip = 0;
        if g.rows == 0 && bands[0].y0 != g.y0 {
            g.y0 = bands[0].y0;
            skip = (32 - g.y0 % 32) % 32;
            skip = skip.min(bands[0].h);
            g.y0 += skip;
        }
        let luma_h = bands[0].h - skip;
        for (p, b) in bands.iter().enumerate() {
            let w = g.widths[p];
            let sy = if p > 0 { g.sub_y } else { 0 };
            let pskip = skip >> sy;
            let at = (g.rows >> sy) * w;
            let h = b.h - pskip;
            let need = at + h * w;
            if g.buf[p].len() < need {
                g.buf[p].resize(need, T::default());
            }
            for i in 0..h {
                let si = (i + pskip) * b.stride;
                g.buf[p][at + i * w..at + (i + 1) * w].copy_from_slice(&b.data[si..si + w]);
            }
        }
        g.rows += luma_h;
        // Process complete 32-row chunks (or everything at the frame end).
        let end = g.y0 + g.rows == g.frame_h;
        let chunk = if end { g.rows } else { g.rows / 32 * 32 };
        if chunk == 0 {
            return Ok(());
        }
        {
            let strides = g.widths.clone();
            let mut refs: Vec<&mut [T]> = g.buf.iter_mut().map(|v| &mut v[..]).collect();
            g.fg.apply_band(&mut refs, &strides, g.y0, chunk, g.frame_w, g.frame_h);
        }
        let sub_y = g.sub_y;
        let y0 = g.y0;
        let mut out_bands = Vec::new();
        let bufs = std::mem::take(&mut g.buf);
        let widths = g.widths.clone();
        for p in 0..bufs.len() {
            let sy = if p > 0 { sub_y } else { 0 };
            let py0 = y0 >> sy;
            let py1 = if end { (y0 + chunk + sy) >> sy } else { (y0 + chunk) >> sy };
            out_bands.push((py0, py1 - py0, widths[p]));
        }
        {
            let bands: Vec<PlaneRows<'_, T>> = out_bands
                .iter()
                .enumerate()
                .map(|(p, &(py0, h, w))| PlaneRows {
                    y0: py0,
                    h,
                    w,
                    data: &bufs[p][..],
                    stride: w,
                })
                .collect();
            self.forward(&bands);
        }
        let g = self.grain.as_mut().unwrap();
        g.buf = bufs;
        // Drop the forwarded rows.
        for (p, b) in g.buf.iter_mut().enumerate() {
            let sy = if p > 0 { sub_y } else { 0 };
            let w = widths[p];
            let drop = out_bands[p].1 * w;
            let keep = ((g.rows >> sy) * w).saturating_sub(drop);
            b.copy_within(drop..drop + keep, 0);
        }
        g.y0 += chunk;
        g.rows -= chunk;
        Ok(())
    }
}
