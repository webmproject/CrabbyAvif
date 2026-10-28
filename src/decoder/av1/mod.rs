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

//! av1: an AV1 keyframe-only decoder for still images (AVIF).
//!
//! ```no_run
//! let data = std::fs::read("image.obu").unwrap();
//! let frame = crabby_avif::decoder::av1::decode(&data, &crabby_avif::decoder::av1::DecodeOptions::default()).unwrap();
//! println!("{}x{}", frame.width, frame.height);
//! ```
//!
//! See `agent_output/DESIGN.md` for the architecture.

// The decoder is 100% safe Rust: unsafe code is forbidden crate-wide and
// cannot be re-enabled locally with `#[allow(unsafe_code)]`. Hot loops
// avoid per-element bounds checks by slicing once up front (e.g.
// `split_at` / `chunks_exact` / fixed-size array conversions) and use
// `std::simd` forms in place of `core::arch` intrinsics.
#![forbid(unsafe_code)]
#![allow(dead_code)]
// Spec-literal code reads better with explicit index loops and the
// spec's if/else chains; DSP kernels take many scalar parameters.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::if_same_then_else
)]

mod bitreader;
mod decode;
pub mod dsp;
mod error;
mod headers;
mod msac;
mod obu;
pub mod pixel;
pub mod scale;
mod sink;
pub mod tables;
pub mod types;

pub use decode::FilterMask;
pub use error::{Error, Result};
pub use sink::Plane;

use decode::{DecodeParams, SbRegion};
use dsp::film_grain::FilmGrain;
use headers::MC_IDENTITY;
use pixel::Pixel;
use sink::{OutputSink, PlaneCrop};

/// Rectangle in luma samples of the (upscaled) frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// Chroma subsampling of the decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaLayout {
    Mono,
    Yuv420,
    Yuv422,
    Yuv444,
}

impl ChromaLayout {
    fn sub(self) -> (usize, usize) {
        match self {
            ChromaLayout::Mono | ChromaLayout::Yuv420 => (1, 1),
            ChromaLayout::Yuv422 => (1, 0),
            ChromaLayout::Yuv444 => (0, 0),
        }
    }
}

/// Options for [`decode`].
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Only decode this region of the frame (luma coordinates). `None`
    /// decodes the whole frame.
    pub crop: Option<Rect>,
    /// Output size (luma). Must not be larger than the crop (or frame)
    /// size; the decoded pixels are downscaled to it on the fly with an
    /// area-average filter. `None` keeps the crop size.
    pub output_size: Option<(usize, usize)>,
    /// Apply film grain synthesis if the stream signals it.
    pub apply_film_grain: bool,
    /// In-loop filters to run (testing aid; keep the default).
    #[doc(hidden)]
    pub filters: FilterMask,
    /// Largest frame (upscaled width x height, in luma samples) that is
    /// decoded; larger frames fail with [`Error::Unsupported`] before
    /// anything is allocated. AV1 allows frames up to 65536x65536, which
    /// a malicious input of a few hundred bytes can request. The default
    /// is [`DEFAULT_MAX_FRAME_PIXELS`] (16384 x 16384, libavif's default
    /// limit).
    pub max_frame_pixels: usize,
    /// Maximum number of threads used for tile decoding. `1` (the
    /// default) decodes on the calling thread; `0` uses
    /// [`std::thread::available_parallelism`].
    pub threads: usize,
}

/// Default value of [`DecodeOptions::max_frame_pixels`].
pub const DEFAULT_MAX_FRAME_PIXELS: usize = 16384 * 16384;

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            crop: None,
            output_size: None,
            apply_film_grain: true,
            filters: FilterMask::default(),
            max_frame_pixels: DEFAULT_MAX_FRAME_PIXELS,
            threads: 1,
        }
    }
}

/// Basic information about the coded image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    /// Frame width (after super-resolution upscaling) and height.
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub layout: ChromaLayout,
    pub color_range_full: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// Decoded samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Planes {
    /// 8-bit samples.
    U8(Vec<Plane<u8>>),
    /// 10 or 12-bit samples.
    U16(Vec<Plane<u16>>),
}

/// A decoded frame (1 plane for monochrome, else Y, U, V).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub layout: ChromaLayout,
    pub planes: Planes,
}

impl Frame {
    /// Writes the planar YUV bytes of [`Frame::to_yuv_bytes`] to `w`
    /// without building the whole buffer in memory first.
    ///
    /// The rows are copied (high bit depth rows converted to little
    /// endian bytes) into one reused [`WRITE_BUF`] byte buffer, which is
    /// written whenever it is full: the kernel then copies from a cache
    /// hot buffer (as with stdio) instead of reading the cold planes
    /// itself, which is slow on arm64 (`copy_from_user`). On other
    /// targets (x86-64: `rep movsb` copies) 8-bit rows are handed to the
    /// writer directly instead (one `writev` per up to [`WRITE_BATCH`]
    /// rows), which saves the user space copy.
    pub fn write_yuv<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        #[cfg(not(target_arch = "aarch64"))]
        if let Planes::U8(ps) = &self.planes {
            for p in ps {
                if p.stride == p.width {
                    w.write_all(&p.samples()[..p.width * p.height])?;
                } else {
                    write_rows(w, (0..p.height).map(|y| p.row(y)))?;
                }
            }
            return Ok(());
        }
        let mut buf = vec![0u8; WRITE_BUF];
        let mut len = 0;
        match &self.planes {
            Planes::U8(ps) => {
                for p in ps {
                    for y in 0..p.height {
                        let mut row = p.row(y);
                        while !row.is_empty() {
                            let n = row.len().min(WRITE_BUF - len);
                            buf[len..len + n].copy_from_slice(&row[..n]);
                            len += n;
                            row = &row[n..];
                            if len == WRITE_BUF {
                                w.write_all(&buf)?;
                                len = 0;
                            }
                        }
                    }
                }
            }
            Planes::U16(ps) => {
                for p in ps {
                    for y in 0..p.height {
                        let mut row = p.row(y);
                        while !row.is_empty() {
                            let n = row.len().min((WRITE_BUF - len) / 2);
                            for (d, v) in buf[len..len + 2 * n]
                                .as_chunks_mut::<2>()
                                .0
                                .iter_mut()
                                .zip(&row[..n])
                            {
                                *d = v.to_le_bytes();
                            }
                            len += 2 * n;
                            row = &row[n..];
                            if len == WRITE_BUF {
                                w.write_all(&buf)?;
                                len = 0;
                            }
                        }
                    }
                }
            }
        }
        if len > 0 {
            w.write_all(&buf[..len])?;
        }
        Ok(())
    }

    /// Planar YUV bytes (16-bit little endian for high bit depth), the
    /// format written by `dav1d -o out.yuv`.
    pub fn to_yuv_bytes(&self) -> Vec<u8> {
        match &self.planes {
            Planes::U8(ps) => {
                let mut out = Vec::with_capacity(ps.iter().map(|p| p.width * p.height).sum());
                for p in ps {
                    if p.stride == p.width {
                        out.extend_from_slice(&p.samples()[..p.width * p.height]);
                    } else {
                        for y in 0..p.height {
                            out.extend_from_slice(p.row(y));
                        }
                    }
                }
                out
            }
            Planes::U16(ps) => {
                let mut out = vec![0u8; ps.iter().map(|p| 2 * p.width * p.height).sum()];
                let mut at = 0;
                for p in ps {
                    for y in 0..p.height {
                        let dst = &mut out[at..at + 2 * p.width];
                        for (d, v) in dst.chunks_exact_mut(2).zip(p.row(y)) {
                            d.copy_from_slice(&v.to_le_bytes());
                        }
                        at += 2 * p.width;
                    }
                }
                out
            }
        }
    }
}

/// Size of the staging buffer of [`Frame::write_yuv`] (64 KiB: small
/// enough to stay in L1/L2, large enough to keep the write calls few).
const WRITE_BUF: usize = 64 * 1024;

/// Rows per vectored write in [`Frame::write_yuv`] (`IOV_MAX` on Linux).
#[cfg(not(target_arch = "aarch64"))]
const WRITE_BATCH: usize = 1024;

/// Writes `rows` back to back to `w`, [`WRITE_BATCH`] rows per
/// `write_vectored` call (retried until each batch is fully written).
#[cfg(not(target_arch = "aarch64"))]
fn write_rows<'a, W: std::io::Write>(
    w: &mut W,
    rows: impl Iterator<Item = &'a [u8]>,
) -> std::io::Result<()> {
    use std::io::{ErrorKind, IoSlice};
    let mut rows = rows.peekable();
    let mut bufs: Vec<IoSlice<'a>> = Vec::with_capacity(WRITE_BATCH);
    while rows.peek().is_some() {
        bufs.clear();
        bufs.extend(rows.by_ref().take(WRITE_BATCH).map(IoSlice::new));
        let mut slices = &mut bufs[..];
        IoSlice::advance_slices(&mut slices, 0);
        while !slices.is_empty() {
            match w.write_vectored(slices) {
                Ok(0) => return Err(ErrorKind::WriteZero.into()),
                Ok(n) => IoSlice::advance_slices(&mut slices, n),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

fn info_from(seq: &headers::SequenceHeader, fh: &headers::FrameHeader) -> ImageInfo {
    let c = &seq.color;
    let layout = if c.mono_chrome {
        ChromaLayout::Mono
    } else if c.subsampling_x && c.subsampling_y {
        ChromaLayout::Yuv420
    } else if c.subsampling_x {
        ChromaLayout::Yuv422
    } else {
        ChromaLayout::Yuv444
    };
    ImageInfo {
        width: fh.upscaled_width,
        height: fh.frame_height,
        bit_depth: c.bit_depth,
        layout,
        color_range_full: c.color_range,
        color_primaries: c.color_primaries,
        transfer_characteristics: c.transfer_characteristics,
        matrix_coefficients: c.matrix_coefficients,
    }
}

/// Parses the headers of the first frame and returns its properties.
pub fn probe(data: &[u8]) -> Result<ImageInfo> {
    let fd = obu::parse_first_frame(data)?;
    Ok(info_from(&fd.seq, &fd.frame))
}

/// One line summary of the coding tools enabled in the frame header
/// (debugging / test aid).
#[doc(hidden)]
pub fn header_summary(data: &[u8]) -> Result<String> {
    let fd = obu::parse_first_frame(data)?;
    let (s, f) = (&fd.seq, &fd.frame);
    Ok(format!(
        "{}x{} coded {} sb128={} tiles={}x{} superres={} intrabc={} sct={} lossless={} qm={} \
         seg={} dq={} dlf={} lf={:?} cdef={} lr={:?} txmode={:?} rtx={} grain={} fi={} ief={}",
        f.upscaled_width,
        f.frame_height,
        f.frame_width,
        s.use_128x128_superblock,
        f.tile_info.cols,
        f.tile_info.rows,
        f.use_superres,
        f.allow_intrabc,
        f.allow_screen_content_tools,
        f.coded_lossless,
        f.quant.using_qmatrix,
        f.seg.enabled,
        f.delta_q_present,
        f.delta_lf_present,
        f.lf.level,
        s.enable_cdef && f.cdef.bits + f.cdef.y_pri_strength[0] + f.cdef.y_sec_strength[0] > 0,
        f.lr.frame_restoration_type,
        f.tx_mode,
        f.reduced_tx_set,
        f.film_grain.apply_grain,
        s.enable_filter_intra,
        s.enable_intra_edge_filter,
    ))
}

/// Decodes the first frame (which must be an intra frame) of a stream of
/// OBUs (low overhead bitstream format, e.g. the payload of an AVIF item).
pub fn decode(data: &[u8], opts: &DecodeOptions) -> Result<Frame> {
    let fd = obu::parse_first_frame(data)?;
    let info = info_from(&fd.seq, &fd.frame);
    if info.width.saturating_mul(info.height) > opts.max_frame_pixels {
        return Err(Error::Unsupported(
            "frame larger than DecodeOptions::max_frame_pixels",
        ));
    }
    if info.bit_depth == 8 {
        let planes = decode_typed::<u8>(&fd, &info, opts)?;
        let (w, h) = (planes[0].width, planes[0].height);
        Ok(Frame {
            width: w,
            height: h,
            bit_depth: 8,
            layout: info.layout,
            planes: Planes::U8(planes),
        })
    } else {
        let planes = decode_typed::<u16>(&fd, &info, opts)?;
        let (w, h) = (planes[0].width, planes[0].height);
        Ok(Frame {
            width: w,
            height: h,
            bit_depth: info.bit_depth,
            layout: info.layout,
            planes: Planes::U16(planes),
        })
    }
}

fn decode_typed<T: Pixel>(
    fd: &obu::FrameData,
    info: &ImageInfo,
    opts: &DecodeOptions,
) -> Result<Vec<Plane<T>>> {
    let crop = opts.crop.unwrap_or(Rect {
        x: 0,
        y: 0,
        width: info.width,
        height: info.height,
    });
    let fits = |pos: usize, len: usize, max: usize| {
        len != 0 && pos.checked_add(len).is_some_and(|end| end <= max)
    };
    if !fits(crop.x, crop.width, info.width) || !fits(crop.y, crop.height, info.height) {
        return Err(Error::InvalidOptions("crop rectangle outside the frame"));
    }
    let (ow, oh) = opts.output_size.unwrap_or((crop.width, crop.height));
    if ow == 0 || oh == 0 || ow > crop.width || oh > crop.height {
        return Err(Error::InvalidOptions(
            "output size must be within 1..=crop size",
        ));
    }
    let np = if info.layout == ChromaLayout::Mono { 1 } else { 3 };
    let (ssx, ssy) = info.layout.sub();
    let mut plane_dims = Vec::new();
    let mut crops = Vec::new();
    let mut out_dims = Vec::new();
    for p in 0..np {
        let (sx, sy) = if p == 0 { (0, 0) } else { (ssx, ssy) };
        plane_dims.push(((info.width + sx) >> sx, (info.height + sy) >> sy));
        let x0 = crop.x >> sx;
        let y0 = crop.y >> sy;
        let x1 = (crop.x + crop.width + sx) >> sx;
        let y1 = (crop.y + crop.height + sy) >> sy;
        crops.push(PlaneCrop {
            x0,
            y0,
            w: x1 - x0,
            h: y1 - y0,
        });
        if opts.output_size.is_some() {
            out_dims.push(((ow + sx) >> sx, (oh + sy) >> sy));
        } else {
            out_dims.push((x1 - x0, y1 - y0));
        }
    }
    let fh = &fd.frame;
    let grain = if opts.apply_film_grain && fh.film_grain.apply_grain {
        let c = &fd.seq.color;
        Some((
            FilmGrain::new(
                &fh.film_grain,
                c.bit_depth,
                c.mono_chrome,
                c.subsampling_x,
                c.subsampling_y,
                c.matrix_coefficients == MC_IDENTITY,
            ),
            ssy,
        ))
    } else {
        None
    };
    let mut sink = OutputSink::<T>::new(&plane_dims, crops, &out_dims, grain, info.bit_depth);
    let region = if opts.crop.is_some() { Some(sb_region(fd, &crop)) } else { None };
    let threads = if opts.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        opts.threads
    };
    let params = DecodeParams {
        filters: opts.filters,
        region,
        threads,
    };
    decode::decode_frame::<T>(fd, &params, &mut sink)?;
    Ok(sink.planes)
}

/// Superblocks needed for a crop rectangle (grown by one superblock for
/// the in-loop filter footprint).
fn sb_region(fd: &obu::FrameData, crop: &Rect) -> SbRegion {
    let fh = &fd.frame;
    let sb_px = if fd.seq.use_128x128_superblock { 128 } else { 64 };
    let sb_cols = fh.mi_cols.div_ceil(sb_px / 4);
    let sb_rows = fh.mi_rows.div_ceil(sb_px / 4);
    // Map upscaled x to coded x (superres) conservatively.
    let to_coded = |x: usize| -> usize { x * fh.frame_width / fh.upscaled_width };
    let x0 = to_coded(crop.x);
    let x1 = to_coded(crop.x + crop.width).max(x0 + 1);
    let col_start = (x0 / sb_px).saturating_sub(1);
    let col_end = (x1.div_ceil(sb_px) + 1).min(sb_cols);
    let row_start = (crop.y / sb_px).saturating_sub(1);
    let row_end = ((crop.y + crop.height).div_ceil(sb_px) + 1).min(sb_rows);
    SbRegion {
        row_end,
        col_start,
        col_end,
        row_start,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_u16(width: usize, height: usize) -> Frame {
        let mut planes = Vec::new();
        for (p, (w, h)) in [
            (width, height),
            (width.div_ceil(2), height.div_ceil(2)),
            (width.div_ceil(2), height.div_ceil(2)),
        ]
        .into_iter()
        .enumerate()
        {
            let mut pl = Plane::<u16>::new(w, h);
            for (i, v) in pl.data.iter_mut().enumerate() {
                *v = ((i * 7 + p * 13) % 4096) as u16;
            }
            planes.push(pl);
        }
        Frame {
            width,
            height,
            bit_depth: 12,
            layout: ChromaLayout::Yuv420,
            planes: Planes::U16(planes),
        }
    }

    #[test]
    fn write_yuv_matches_to_yuv_bytes() {
        for (w, h) in [(1, 1), (17, 9), (300, 260)] {
            let f = frame_u16(w, h);
            let mut out = Vec::new();
            f.write_yuv(&mut out).unwrap();
            assert_eq!(out, f.to_yuv_bytes());
            let Planes::U16(ps) = &f.planes else {
                unreachable!()
            };
            let p8: Vec<Plane<u8>> = ps
                .iter()
                .map(|p| Plane {
                    width: p.width,
                    height: p.height,
                    stride: p.stride,
                    offset: p.offset,
                    data: p.data.iter().map(|&v| v as u8).collect(),
                })
                .collect();
            let f8 = Frame {
                planes: Planes::U8(p8),
                bit_depth: 8,
                ..f
            };
            let mut out = Vec::new();
            f8.write_yuv(&mut out).unwrap();
            assert_eq!(out, f8.to_yuv_bytes());
            assert_eq!(out.len(), w * h + 2 * w.div_ceil(2) * h.div_ceil(2));
        }
    }

    /// A writer that accepts at most `max` bytes per call (and sometimes
    /// fails with `Interrupted`), like a short `writev`.
    struct ShortWriter {
        out: Vec<u8>,
        max: usize,
        calls: usize,
    }

    impl std::io::Write for ShortWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.write_vectored(&[std::io::IoSlice::new(buf)])
        }

        fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
            self.calls += 1;
            if self.calls % 5 == 0 {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let mut n = 0;
            for b in bufs {
                let take = b.len().min(self.max - n);
                self.out.extend_from_slice(&b[..take]);
                n += take;
                if n == self.max {
                    break;
                }
            }
            Ok(n)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Planes with padding (`stride > width`, as handed out by the
    /// decoder) take the vectored row path; short writes are resumed.
    #[test]
    fn write_yuv_strided_rows() {
        for (w, h, pad) in [
            (1usize, 1usize, 1usize),
            (17, 9, 64),
            (300, 1100, 7),
            (2, 3000, 0),
        ] {
            let planes: Vec<Plane<u8>> = [
                (w, h),
                (w.div_ceil(2), h.div_ceil(2)),
                (w.div_ceil(2), h.div_ceil(2)),
            ]
            .into_iter()
            .enumerate()
            .map(|(p, (pw, ph))| {
                let stride = pw + pad;
                let data = (0..stride * ph).map(|i| (i * 7 + p * 13) as u8).collect();
                Plane {
                    width: pw,
                    height: ph,
                    stride,
                    offset: 0,
                    data,
                }
            })
            .collect();
            let want: Vec<u8> = planes
                .iter()
                .flat_map(|p| (0..p.height).flat_map(move |y| p.row(y).to_vec()))
                .collect();
            let f = Frame {
                width: w,
                height: h,
                bit_depth: 8,
                layout: ChromaLayout::Yuv420,
                planes: Planes::U8(planes),
            };
            assert_eq!(f.to_yuv_bytes(), want);
            let mut out = Vec::new();
            f.write_yuv(&mut out).unwrap();
            assert_eq!(out, want);
            for max in [1, 5, 4096, 1 << 20] {
                let mut sw = ShortWriter {
                    out: Vec::new(),
                    max,
                    calls: 0,
                };
                f.write_yuv(&mut sw).unwrap();
                assert!(sw.out == want, "{w}x{h} pad {pad} max {max}");
            }
        }
    }
}
