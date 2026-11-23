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

#![forbid(unsafe_code)]
#![allow(dead_code)] // Used in follow-up CLI commands.

use crabby_avif::gainmap::{GainMap, GainMapMetadata};
use crabby_avif::image;
use crabby_avif::reformat::rgb;
use crabby_avif::utils::{Fraction, UFraction};
use crabby_avif::{
    AvifError, AvifResult, ColorPrimaries, ContentLightLevelInformation, MatrixCoefficients,
    PixelFormat, TransferCharacteristics,
};

const SDR_WHITE_NITS: f32 = 203.0;
const PQ_MAX_NITS: f32 = 10000.0;
const HLG_PEAK_LUMINANCE_NITS: f32 = 1000.0;
const EPSILON_F64: f64 = 1e-12;
const EPSILON_F32: f32 = 1e-10;

fn try_alloc_vec<T: Clone>(elem: T, len: usize) -> AvifResult<Vec<T>> {
    let mut vec = Vec::new();
    vec.try_reserve_exact(len)
        .map_err(|_| AvifError::OutOfMemory)?;
    vec.resize(len, elem);
    Ok(vec)
}

fn try_clone_bytes(src: &[u8]) -> AvifResult<Vec<u8>> {
    let mut dst = Vec::new();
    dst.try_reserve_exact(src.len())
        .map_err(|_| AvifError::OutOfMemory)?;
    dst.extend_from_slice(src);
    Ok(dst)
}

fn nan_safe_clamp(val: f32) -> f32 {
    0.0f32.max(val).min(1.0)
}

fn fraction_to_f32(f: Fraction) -> f32 {
    if f.1 == 0 {
        0.0
    } else {
        f.0 as f32 / f.1 as f32
    }
}

fn ufraction_to_f32(f: UFraction) -> f32 {
    if f.1 == 0 {
        0.0
    } else {
        f.0 as f32 / f.1 as f32
    }
}

fn lerp(a: f32, b: f32, w: f32) -> f32 {
    (1.0 - w) * a + w * b
}

fn to_linear_709(gamma: f32) -> f32 {
    if gamma < 0.0 {
        0.0
    } else if gamma < 4.5 * 0.018_053_97 {
        gamma / 4.5
    } else if gamma < 1.0 {
        ((gamma + 0.099_296_83) / 1.099_296_8).powf(1.0 / 0.45)
    } else {
        1.0
    }
}

fn to_gamma_709(linear: f32) -> f32 {
    if linear < 0.0 {
        0.0
    } else if linear < 0.018_053_97 {
        linear * 4.5
    } else if linear < 1.0 {
        1.099_296_8 * linear.powf(0.45) - 0.099_296_83
    } else {
        1.0
    }
}

fn to_linear_srgb(gamma: f32) -> f32 {
    if gamma < 0.0 {
        0.0
    } else if gamma < 12.92 * 0.003_041_282_5 {
        gamma / 12.92
    } else if gamma < 1.0 {
        ((gamma + 0.055_010_72) / 1.055_010_7).powf(2.4)
    } else {
        1.0
    }
}

fn to_gamma_srgb(linear: f32) -> f32 {
    if linear < 0.0 {
        0.0
    } else if linear < 0.003_041_282_5 {
        linear * 12.92
    } else if linear < 1.0 {
        1.055_010_7 * linear.powf(1.0 / 2.4) - 0.055_010_72
    } else {
        1.0
    }
}

fn to_linear_pq(gamma: f32) -> f32 {
    if gamma > 0.0 {
        let pow_gamma = gamma.powf(1.0 / 78.84375);
        let num = (pow_gamma - 0.8359375).max(0.0);
        let den = (18.851563 - 18.6875 * pow_gamma).max(f32::MIN_POSITIVE);
        let linear = (num / den).powf(1.0 / 0.159_301_76);
        linear * PQ_MAX_NITS / SDR_WHITE_NITS
    } else {
        0.0
    }
}

fn to_gamma_pq(linear: f32) -> f32 {
    if linear > 0.0 {
        let scaled = (linear * SDR_WHITE_NITS / PQ_MAX_NITS).clamp(0.0, 1.0);
        let pow_linear = scaled.powf(0.159_301_76);
        let num = 0.1640625 * pow_linear - 0.1640625;
        let den = 1.0 + 18.6875 * pow_linear;
        (1.0 + num / den).powf(78.84375)
    } else {
        0.0
    }
}

fn to_linear_hlg(gamma: f32) -> f32 {
    if gamma < 0.0 {
        return 0.0;
    }
    let linear = if gamma <= 0.5 {
        ((gamma * gamma) * (1.0 / 3.0)).powf(1.2)
    } else {
        ((((gamma - 0.559_910_7) / 0.178_832_77).exp() + 0.284_668_92) / 12.0).powf(1.2)
    };
    linear * HLG_PEAK_LUMINANCE_NITS / SDR_WHITE_NITS
}

fn to_gamma_hlg(linear: f32) -> f32 {
    let scaled = (linear * SDR_WHITE_NITS / HLG_PEAK_LUMINANCE_NITS).clamp(0.0, 1.0);
    let inv_ootf = scaled.powf(1.0 / 1.2);
    if inv_ootf <= 0.0 {
        0.0
    } else if inv_ootf <= 1.0 / 12.0 {
        (3.0 * inv_ootf).sqrt()
    } else {
        0.178_832_77 * (12.0 * inv_ootf - 0.284_668_92).ln() + 0.559_910_7
    }
}

fn to_linear_smpte240(gamma: f32) -> f32 {
    if gamma < 0.0 {
        0.0
    } else if gamma < 4.0 * 0.022_821_585 {
        gamma / 4.0
    } else if gamma < 1.0 {
        ((gamma + 0.111_572_2) / 1.111_572_2).powf(1.0 / 0.45)
    } else {
        1.0
    }
}

fn to_gamma_smpte240(linear: f32) -> f32 {
    if linear < 0.0 {
        0.0
    } else if linear < 0.022_821_585 {
        linear * 4.0
    } else if linear < 1.0 {
        1.111_572_2 * linear.powf(0.45) - 0.111_572_2
    } else {
        1.0
    }
}

fn to_linear_log100(gamma: f32) -> f32 {
    if gamma <= 0.0 {
        0.01 / 2.0
    } else {
        10.0f32.powf(2.0 * (gamma.min(1.0) - 1.0))
    }
}

fn to_gamma_log100(linear: f32) -> f32 {
    if linear <= 0.01 {
        0.0
    } else {
        1.0 + linear.min(1.0).log10() / 2.0
    }
}

fn to_linear_log100_sqrt10(gamma: f32) -> f32 {
    if gamma <= 0.0 {
        0.003_162_277_6 / 2.0
    } else {
        10.0f32.powf(2.5 * (gamma.min(1.0) - 1.0))
    }
}

fn to_gamma_log100_sqrt10(linear: f32) -> f32 {
    if linear <= 0.003_162_277_6 {
        0.0
    } else {
        1.0 + linear.min(1.0).log10() / 2.5
    }
}

fn to_linear_iec61966(gamma: f32) -> f32 {
    if gamma < -4.5 * 0.018_053_97 {
        -((gamma - 0.099_296_83) / -1.099_296_8).powf(1.0 / 0.45)
    } else if gamma < 4.5 * 0.018_053_97 {
        gamma / 4.5
    } else {
        ((gamma + 0.099_296_83) / 1.099_296_8).powf(1.0 / 0.45)
    }
}

fn to_gamma_iec61966(linear: f32) -> f32 {
    if linear < -0.018_053_97 {
        -1.099_296_8 * (-linear).powf(0.45) + 0.099_296_83
    } else if linear < 0.018_053_97 {
        linear * 4.5
    } else {
        1.099_296_8 * linear.powf(0.45) - 0.099_296_83
    }
}

fn to_linear_bt1361(gamma: f32) -> f32 {
    if gamma < -0.25 {
        -0.25
    } else if gamma < 0.0 {
        ((gamma - 0.024_824_206) / -0.274_824_2).powf(1.0 / 0.45) / -4.0
    } else if gamma < 4.5 * 0.018_053_97 {
        gamma / 4.5
    } else if gamma < 1.0 {
        ((gamma + 0.099_296_83) / 1.099_296_8).powf(1.0 / 0.45)
    } else {
        1.0
    }
}

fn to_gamma_bt1361(linear: f32) -> f32 {
    if linear < -0.25 {
        -0.25
    } else if linear < 0.0 {
        -0.274_824_2 * (-4.0 * linear).powf(0.45) + 0.024_824_206
    } else if linear < 0.018_053_97 {
        linear * 4.5
    } else if linear < 1.0 {
        1.099_296_8 * linear.powf(0.45) - 0.099_296_83
    } else {
        1.0
    }
}

pub fn gamma_to_linear_fn(tc: TransferCharacteristics) -> fn(f32) -> f32 {
    match tc {
        TransferCharacteristics::Bt470m => |g| g.clamp(0.0, 1.0).powf(2.2),
        TransferCharacteristics::Bt470bg => |g| g.clamp(0.0, 1.0).powf(2.8),
        TransferCharacteristics::Smpte240 => to_linear_smpte240,
        TransferCharacteristics::Linear => |g| g.clamp(0.0, 1.0),
        TransferCharacteristics::Log100 => to_linear_log100,
        TransferCharacteristics::Log100Sqrt10 => to_linear_log100_sqrt10,
        TransferCharacteristics::Iec61966 => to_linear_iec61966,
        TransferCharacteristics::Bt1361 => to_linear_bt1361,
        TransferCharacteristics::Srgb => to_linear_srgb,
        TransferCharacteristics::Pq => to_linear_pq,
        TransferCharacteristics::Smpte428 => |g| g.clamp(0.0, 1.0).powf(2.6) / 0.916_555_3,
        TransferCharacteristics::Hlg => to_linear_hlg,
        // Bt709, Bt601, Bt2020_10bit, Bt2020_12bit, and unknown TCs default to Rec. 709.
        _ => to_linear_709,
    }
}

pub fn linear_to_gamma_fn(tc: TransferCharacteristics) -> fn(f32) -> f32 {
    match tc {
        TransferCharacteristics::Bt470m => |l| l.clamp(0.0, 1.0).powf(1.0 / 2.2),
        TransferCharacteristics::Bt470bg => |l| l.clamp(0.0, 1.0).powf(1.0 / 2.8),
        TransferCharacteristics::Smpte240 => to_gamma_smpte240,
        TransferCharacteristics::Linear => |l| l.clamp(0.0, 1.0),
        TransferCharacteristics::Log100 => to_gamma_log100,
        TransferCharacteristics::Log100Sqrt10 => to_gamma_log100_sqrt10,
        TransferCharacteristics::Iec61966 => to_gamma_iec61966,
        TransferCharacteristics::Bt1361 => to_gamma_bt1361,
        TransferCharacteristics::Srgb => to_gamma_srgb,
        TransferCharacteristics::Pq => to_gamma_pq,
        TransferCharacteristics::Smpte428 => {
            |l| (0.916_555_3 * l.max(0.0)).min(1.0).powf(1.0 / 2.6)
        }
        TransferCharacteristics::Hlg => to_gamma_hlg,
        // Bt709, Bt601, Bt2020_10bit, Bt2020_12bit, and unknown TCs default to Rec. 709.
        _ => to_gamma_709,
    }
}

fn primaries_or_default(cp: ColorPrimaries) -> AvifResult<[f32; 8]> {
    cp.values()
        .or_else(|| ColorPrimaries::Srgb.values())
        .ok_or(AvifError::NotImplemented)
}

fn xy_to_xyz(xy: [f32; 2]) -> Option<[f64; 3]> {
    if f64::from(xy[1]).abs() < EPSILON_F64 {
        return None;
    }
    let factor = 1.0 / f64::from(xy[1]);
    Some([
        f64::from(xy[0]) * factor,
        1.0,
        (1.0 - f64::from(xy[0]) - f64::from(xy[1])) * factor,
    ])
}

fn mat_inv(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < EPSILON_F64 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det,
            (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det,
        ],
    ])
}

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn vec_mul(m: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2],
        m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2],
        m[2][0] * x[0] + m[2][1] * x[1] + m[2][2] * x[2],
    ]
}

fn mat_diag(d: [f64; 3]) -> [[f64; 3]; 3] {
    [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
}

const BRADFORD: [[f64; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

const LMS_D50: [f64; 3] = [0.996284, 1.02043, 0.818644];

fn compute_rgb_to_xyz_d50_matrix(cp: ColorPrimaries) -> AvifResult<[[f64; 3]; 3]> {
    let p = primaries_or_default(cp)?;
    let white_xyz = xy_to_xyz([p[6], p[7]]).ok_or(AvifError::NotImplemented)?;
    let rgb_primaries = [
        [f64::from(p[0]), f64::from(p[2]), f64::from(p[4])],
        [f64::from(p[1]), f64::from(p[3]), f64::from(p[5])],
        [
            1.0 - f64::from(p[0]) - f64::from(p[1]),
            1.0 - f64::from(p[2]) - f64::from(p[3]),
            1.0 - f64::from(p[4]) - f64::from(p[5]),
        ],
    ];
    let rgb_primaries_inv = mat_inv(rgb_primaries).ok_or(AvifError::NotImplemented)?;
    let rgb_coeffs = vec_mul(rgb_primaries_inv, white_xyz);
    let rgb_xyz = mat_mul(rgb_primaries, mat_diag(rgb_coeffs));

    let mut lms = vec_mul(BRADFORD, white_xyz);
    for i in 0..3 {
        if lms[i].abs() < EPSILON_F64 {
            return Err(AvifError::NotImplemented);
        }
        lms[i] = LMS_D50[i] / lms[i];
    }
    let bradford_inv = mat_inv(BRADFORD).ok_or(AvifError::NotImplemented)?;
    let adaptation = mat_mul(bradford_inv, mat_mul(mat_diag(lms), BRADFORD));
    Ok(mat_mul(adaptation, rgb_xyz))
}

pub fn compute_rgb_to_rgb_matrix(
    src: ColorPrimaries,
    dst: ColorPrimaries,
) -> AvifResult<[[f64; 3]; 3]> {
    let src_to_xyz = compute_rgb_to_xyz_d50_matrix(src)?;
    let dst_to_xyz = compute_rgb_to_xyz_d50_matrix(dst)?;
    let xyz_to_dst = mat_inv(dst_to_xyz).ok_or(AvifError::NotImplemented)?;
    Ok(mat_mul(xyz_to_dst, src_to_xyz))
}

fn convert_linear_rgb(rgb_pixel: &mut [f32; 3], coeffs: &[[f64; 3]; 3]) {
    let res = vec_mul(
        *coeffs,
        [
            f64::from(rgb_pixel[0]),
            f64::from(rgb_pixel[1]),
            f64::from(rgb_pixel[2]),
        ],
    );
    rgb_pixel[0] = res[0] as f32;
    rgb_pixel[1] = res[1] as f32;
    rgb_pixel[2] = res[2] as f32;
}

fn compute_y_coeffs(cp: ColorPrimaries) -> AvifResult<[f32; 3]> {
    let p = primaries_or_default(cp)?;
    let (rx, ry, gx, gy, bx, by, wx, wy) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    let rz = 1.0 - (rx + ry);
    let gz = 1.0 - (gx + gy);
    let bz = 1.0 - (bx + by);
    let wz = 1.0 - (wx + wy);
    let den = wy * (rx * (gy * bz - by * gz) + gx * (by * rz - ry * bz) + bx * (ry * gz - gy * rz));
    if den.abs() < EPSILON_F32 {
        return Err(AvifError::NotImplemented);
    }
    let kr = (ry
        * (wx * (gy * bz - by * gz) + wy * (bx * gz - gx * bz) + wz * (gx * by - bx * gy)))
        / den;
    let kb = (by
        * (wx * (ry * gz - gy * rz) + wy * (gx * rz - rx * gz) + wz * (rx * gy - gx * ry)))
        / den;
    Ok([kr, 1.0 - kr - kb, kb])
}

#[derive(Clone, Copy)]
struct RgbLayout {
    channels: usize,
    r: usize,
    g: usize,
    b: usize,
    a: usize,
    has_alpha: bool,
    is_gray: bool,
    max_channel: f32,
    is_16bit: bool,
}

impl RgbLayout {
    fn from_image(rgb: &rgb::Image) -> AvifResult<Self> {
        let channels = rgb.channel_count() as usize;
        if channels == 0 || rgb.is_float || !matches!(rgb.depth, 8 | 10 | 12 | 16) {
            return Err(AvifError::NotImplemented);
        }
        let is_gray = matches!(
            rgb.format,
            rgb::Format::Gray | rgb::Format::GrayA | rgb::Format::AGray
        );
        Ok(Self {
            channels,
            r: rgb.format.r_offset(),
            g: rgb.format.g_offset(),
            b: rgb.format.b_offset(),
            a: rgb.format.alpha_offset(),
            has_alpha: rgb.format.has_alpha(),
            is_gray,
            max_channel: f32::from(rgb.max_channel()),
            is_16bit: rgb.depth > 8,
        })
    }
}

fn read_rgba_row(
    rgb: &rgb::Image,
    layout: &RgbLayout,
    y: u32,
    out: &mut [[f32; 4]],
) -> AvifResult<()> {
    let inv_max = 1.0 / layout.max_channel;
    let map_px = |r: f32, g: f32, b: f32, a: f32| -> [f32; 4] {
        if layout.is_gray {
            [r * inv_max, r * inv_max, r * inv_max, a * inv_max]
        } else {
            [r * inv_max, g * inv_max, b * inv_max, a * inv_max]
        }
    };
    if layout.is_16bit {
        let row = rgb.row16(y)?;
        let mut chunks = row.chunks_exact(layout.channels);
        for pixel in out.iter_mut() {
            let chunk = chunks.next().ok_or(AvifError::InvalidArgument)?;
            let r = f32::from(chunk[layout.r]);
            let (g, b) = if layout.is_gray {
                (r, r)
            } else {
                (f32::from(chunk[layout.g]), f32::from(chunk[layout.b]))
            };
            let a = if layout.has_alpha { f32::from(chunk[layout.a]) } else { layout.max_channel };
            *pixel = map_px(r, g, b, a);
        }
    } else {
        let row = rgb.row(y)?;
        let mut chunks = row.chunks_exact(layout.channels);
        for pixel in out.iter_mut() {
            let chunk = chunks.next().ok_or(AvifError::InvalidArgument)?;
            let r = f32::from(chunk[layout.r]);
            let (g, b) = if layout.is_gray {
                (r, r)
            } else {
                (f32::from(chunk[layout.g]), f32::from(chunk[layout.b]))
            };
            let a = if layout.has_alpha { f32::from(chunk[layout.a]) } else { layout.max_channel };
            *pixel = map_px(r, g, b, a);
        }
    }
    Ok(())
}

fn write_rgba_row(
    rgb: &mut rgb::Image,
    layout: &RgbLayout,
    y: u32,
    src: &[[f32; 4]],
) -> AvifResult<()> {
    let max_ch = layout.max_channel;
    if layout.is_16bit {
        let row = rgb.row16_mut(y)?;
        let mut chunks = row.chunks_exact_mut(layout.channels);
        for pixel in src {
            let chunk = chunks.next().ok_or(AvifError::InvalidArgument)?;
            chunk[layout.r] = (nan_safe_clamp(pixel[0]) * max_ch).round() as u16;
            if !layout.is_gray {
                chunk[layout.g] = (nan_safe_clamp(pixel[1]) * max_ch).round() as u16;
                chunk[layout.b] = (nan_safe_clamp(pixel[2]) * max_ch).round() as u16;
            }
            if layout.has_alpha {
                chunk[layout.a] = (nan_safe_clamp(pixel[3]) * max_ch).round() as u16;
            }
        }
    } else {
        let row = rgb.row_mut(y)?;
        let mut chunks = row.chunks_exact_mut(layout.channels);
        for pixel in src {
            let chunk = chunks.next().ok_or(AvifError::InvalidArgument)?;
            chunk[layout.r] = (nan_safe_clamp(pixel[0]) * max_ch).round() as u8;
            if !layout.is_gray {
                chunk[layout.g] = (nan_safe_clamp(pixel[1]) * max_ch).round() as u8;
                chunk[layout.b] = (nan_safe_clamp(pixel[2]) * max_ch).round() as u8;
            }
            if layout.has_alpha {
                chunk[layout.a] = (nan_safe_clamp(pixel[3]) * max_ch).round() as u8;
            }
        }
    }
    Ok(())
}

pub fn validate_gainmap_metadata(metadata: &GainMapMetadata) -> AvifResult<()> {
    for i in 0..3 {
        if metadata.min[i].1 == 0
            || metadata.max[i].1 == 0
            || metadata.gamma[i].1 == 0
            || metadata.base_offset[i].1 == 0
            || metadata.alternate_offset[i].1 == 0
        {
            return Err(AvifError::InvalidArgument);
        }
        if i64::from(metadata.max[i].0) * i64::from(metadata.min[i].1)
            < i64::from(metadata.min[i].0) * i64::from(metadata.max[i].1)
        {
            return Err(AvifError::InvalidArgument);
        }
        if metadata.gamma[i].0 == 0 {
            return Err(AvifError::InvalidArgument);
        }
    }
    if metadata.base_hdr_headroom.1 == 0 || metadata.alternate_hdr_headroom.1 == 0 {
        return Err(AvifError::InvalidArgument);
    }
    Ok(())
}

fn gainmap_weight(hdr_headroom: f32, metadata: &GainMapMetadata) -> f32 {
    let base_headroom = ufraction_to_f32(metadata.base_hdr_headroom);
    let alt_headroom = ufraction_to_f32(metadata.alternate_hdr_headroom);
    if base_headroom == alt_headroom {
        return 0.0;
    }
    let w = ((hdr_headroom - base_headroom) / (alt_headroom - base_headroom)).clamp(0.0, 1.0);
    if alt_headroom < base_headroom {
        -w
    } else {
        w
    }
}

pub fn apply_gainmap(
    base_image: &image::Image,
    gainmap: &GainMap,
    hdr_headroom: f32,
    output_color_primaries: ColorPrimaries,
    output_transfer_characteristics: TransferCharacteristics,
    output_depth: u8,
) -> AvifResult<(rgb::Image, ContentLightLevelInformation)> {
    if !hdr_headroom.is_finite() || hdr_headroom < 0.0 {
        return Err(AvifError::InvalidArgument);
    }
    if !base_image.icc.is_empty() || !gainmap.alt_icc.is_empty() {
        return Err(AvifError::NotImplemented);
    }
    if base_image.width == 0
        || base_image.height == 0
        || !base_image.has_plane(image::Plane::Y)
        || gainmap.image.width == 0
        || gainmap.image.height == 0
        || !gainmap.image.has_plane(image::Plane::Y)
    {
        return Err(AvifError::InvalidArgument);
    }
    if !matches!(base_image.depth, 8 | 10 | 12 | 16)
        || !matches!(gainmap.image.depth, 8 | 10 | 12 | 16)
        || !matches!(output_depth, 8 | 10 | 12 | 16)
    {
        return Err(AvifError::NotImplemented);
    }
    validate_gainmap_metadata(&gainmap.metadata)?;

    let mut base_rgb = rgb::Image::create_from_yuv(base_image);
    base_rgb.format = if base_image.has_alpha() { rgb::Format::Rgba } else { rgb::Format::Rgb };
    base_rgb.allocate()?;
    base_rgb.convert_from_yuv(base_image)?;

    let width = base_rgb.width;
    let height = base_rgb.height;
    let premultiply_alpha = base_image.has_alpha() && base_image.alpha_premultiplied;
    let mut tone_mapped = rgb::Image {
        width,
        height,
        depth: output_depth,
        format: base_rgb.format,
        premultiply_alpha,
        ..Default::default()
    };
    tone_mapped.allocate()?;

    let base_cp = base_image.color_primaries;
    let base_tc = base_image.transfer_characteristics;
    let weight = gainmap_weight(hdr_headroom, &gainmap.metadata);

    let gamma_to_linear = gamma_to_linear_fn(base_tc);
    let linear_to_gamma = linear_to_gamma_fn(output_transfer_characteristics);

    let base_layout = RgbLayout::from_image(&base_rgb)?;
    let out_layout = RgbLayout::from_image(&tone_mapped)?;
    let width_usize = usize::try_from(width).map_err(|_| AvifError::OutOfMemory)?;
    let mut base_row = try_alloc_vec([0.0f32; 4], width_usize)?;
    let mut out_row = try_alloc_vec([0.0f32; 4], width_usize)?;

    if weight == 0.0 {
        let primaries_differ = base_cp != output_color_primaries;
        let conv = if primaries_differ {
            Some(compute_rgb_to_rgb_matrix(base_cp, output_color_primaries)?)
        } else {
            None
        };
        let mut rgb_max_linear = 0.0f32;
        let mut rgb_sum_linear = 0.0f32;
        for y in 0..height {
            read_rgba_row(&base_rgb, &base_layout, y, &mut base_row)?;
            for (dst, src) in out_row.iter_mut().zip(base_row.iter()) {
                let mut rgb_vals = [src[0], src[1], src[2]];
                if output_transfer_characteristics != base_tc || primaries_differ {
                    for ch in &mut rgb_vals {
                        *ch = gamma_to_linear(*ch);
                    }
                    if let Some(m) = &conv {
                        convert_linear_rgb(&mut rgb_vals, m);
                    }
                    let px_max = rgb_vals[0].max(rgb_vals[1]).max(rgb_vals[2]);
                    rgb_max_linear = rgb_max_linear.max(px_max);
                    rgb_sum_linear += px_max;
                    for ch in &mut rgb_vals {
                        *ch = nan_safe_clamp(linear_to_gamma(*ch));
                    }
                } else {
                    let lin = [
                        gamma_to_linear(src[0]),
                        gamma_to_linear(src[1]),
                        gamma_to_linear(src[2]),
                    ];
                    let px_max = lin[0].max(lin[1]).max(lin[2]);
                    rgb_max_linear = rgb_max_linear.max(px_max);
                    rgb_sum_linear += px_max;
                }
                let a = src[3];
                if premultiply_alpha {
                    let a_clamped = nan_safe_clamp(a);
                    for ch in &mut rgb_vals {
                        *ch *= a_clamped;
                    }
                }
                *dst = [rgb_vals[0], rgb_vals[1], rgb_vals[2], a];
            }
            write_rgba_row(&mut tone_mapped, &out_layout, y, &out_row)?;
        }
        let max_cll = (rgb_max_linear * SDR_WHITE_NITS)
            .round()
            .clamp(0.0, f32::from(u16::MAX)) as u16;
        let max_pall = ((rgb_sum_linear / (width as f32 * height as f32)) * SDR_WHITE_NITS)
            .round()
            .clamp(0.0, f32::from(u16::MAX)) as u16;
        return Ok((
            tone_mapped,
            ContentLightLevelInformation { max_cll, max_pall },
        ));
    }

    let math_cp = if gainmap.metadata.use_base_color_space
        || gainmap.alt_color_primaries == ColorPrimaries::Unspecified
    {
        base_cp
    } else {
        gainmap.alt_color_primaries
    };
    let in_conv = if base_cp != math_cp {
        Some(compute_rgb_to_rgb_matrix(base_cp, math_cp)?)
    } else {
        None
    };
    let out_conv = if math_cp != output_color_primaries {
        Some(compute_rgb_to_rgb_matrix(math_cp, output_color_primaries)?)
    } else {
        None
    };

    let mut scaled_gm;
    let gm_image = if gainmap.image.width != width || gainmap.image.height != height {
        scaled_gm = gainmap.image.try_deep_clone()?;
        scaled_gm.scale(width, height)?;
        &scaled_gm
    } else {
        &gainmap.image
    };

    let mut gm_rgb = rgb::Image::create_from_yuv(gm_image);
    gm_rgb.format = rgb::Format::Rgb;
    gm_rgb.allocate()?;
    gm_rgb.convert_from_yuv(gm_image)?;
    let gm_layout = RgbLayout::from_image(&gm_rgb)?;
    let mut gm_row = try_alloc_vec([0.0f32; 4], width_usize)?;

    let gamma_inv = [
        1.0 / ufraction_to_f32(gainmap.metadata.gamma[0]),
        1.0 / ufraction_to_f32(gainmap.metadata.gamma[1]),
        1.0 / ufraction_to_f32(gainmap.metadata.gamma[2]),
    ];
    let gm_min = [
        fraction_to_f32(gainmap.metadata.min[0]),
        fraction_to_f32(gainmap.metadata.min[1]),
        fraction_to_f32(gainmap.metadata.min[2]),
    ];
    let gm_max = [
        fraction_to_f32(gainmap.metadata.max[0]),
        fraction_to_f32(gainmap.metadata.max[1]),
        fraction_to_f32(gainmap.metadata.max[2]),
    ];
    let base_offset = [
        fraction_to_f32(gainmap.metadata.base_offset[0]),
        fraction_to_f32(gainmap.metadata.base_offset[1]),
        fraction_to_f32(gainmap.metadata.base_offset[2]),
    ];
    let alt_offset = [
        fraction_to_f32(gainmap.metadata.alternate_offset[0]),
        fraction_to_f32(gainmap.metadata.alternate_offset[1]),
        fraction_to_f32(gainmap.metadata.alternate_offset[2]),
    ];

    let mut rgb_max_linear = 0.0f32;
    let mut rgb_sum_linear = 0.0f32;
    for y in 0..height {
        read_rgba_row(&base_rgb, &base_layout, y, &mut base_row)?;
        read_rgba_row(&gm_rgb, &gm_layout, y, &mut gm_row)?;
        for x in 0..(width as usize) {
            let mut lin = [
                gamma_to_linear(base_row[x][0]),
                gamma_to_linear(base_row[x][1]),
                gamma_to_linear(base_row[x][2]),
            ];
            if let Some(m) = &in_conv {
                convert_linear_rgb(&mut lin, m);
            }
            let mut px_max = 0.0f32;
            let mut mapped = [0.0f32; 3];
            for c in 0..3 {
                let gm_log2 = lerp(gm_min[c], gm_max[c], gm_row[x][c].powf(gamma_inv[c]));
                let v = (lin[c] + base_offset[c]) * (gm_log2 * weight).exp2() - alt_offset[c];
                rgb_max_linear = rgb_max_linear.max(v);
                px_max = px_max.max(v);
                mapped[c] = v;
            }
            if let Some(m) = &out_conv {
                convert_linear_rgb(&mut mapped, m);
            }
            for ch in &mut mapped {
                if !ch.is_finite() {
                    return Err(AvifError::InvalidToneMappedImage(
                        "Degenerate gain map parameters produce non-finite value".into(),
                    ));
                }
                *ch = nan_safe_clamp(linear_to_gamma(*ch));
            }
            let a = base_row[x][3];
            if premultiply_alpha {
                let a_clamped = nan_safe_clamp(a);
                for ch in &mut mapped {
                    *ch *= a_clamped;
                }
            }
            rgb_sum_linear += px_max;
            out_row[x] = [mapped[0], mapped[1], mapped[2], a];
        }
        write_rgba_row(&mut tone_mapped, &out_layout, y, &out_row)?;
    }

    let max_cll = (rgb_max_linear * SDR_WHITE_NITS)
        .round()
        .clamp(0.0, f32::from(u16::MAX)) as u16;
    let avg_linear = rgb_sum_linear / (width as f32 * height as f32);
    let max_pall = (avg_linear * SDR_WHITE_NITS)
        .round()
        .clamp(0.0, f32::from(u16::MAX)) as u16;
    Ok((
        tone_mapped,
        ContentLightLevelInformation { max_cll, max_pall },
    ))
}

fn find_min_max_without_outliers(values: &[f32]) -> AvifResult<(f32, f32)> {
    let Some((&first, rest)) = values.split_first() else {
        return Ok((0.0, 0.0));
    };
    if !first.is_finite() {
        return Err(AvifError::InvalidArgument);
    }
    let bucket_size = 0.01f32;
    let max_outliers_ratio = 0.001f32;
    let max_outliers_each_side =
        ((values.len() as f32) * max_outliers_ratio / 2.0).round() as usize;

    let mut min_v = first;
    let mut max_v = first;
    for &v in rest {
        if !v.is_finite() {
            return Err(AvifError::InvalidArgument);
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        return Err(AvifError::InvalidArgument);
    }
    let mut range_min = min_v;
    let mut range_max = max_v;
    if (max_v - min_v) <= (bucket_size * 2.0) || max_outliers_each_side == 0 {
        return Ok((range_min, range_max));
    }

    let num_buckets = (((max_v - min_v) / bucket_size).ceil() as usize).clamp(1, 10000);
    let mut histogram = try_alloc_vec(0usize, num_buckets)?;
    for &v in values {
        let clamped = v.clamp(min_v, max_v);
        let idx = (((clamped - min_v) / (max_v - min_v) * (num_buckets as f32)).round() as usize)
            .min(num_buckets - 1);
        if let Some(slot) = histogram.get_mut(idx) {
            *slot += 1;
        }
    }

    let mut left_outliers = 0usize;
    for (i, &count) in histogram.iter().enumerate() {
        left_outliers += count;
        if left_outliers > max_outliers_each_side {
            break;
        }
        if count == 0 {
            range_min = ((i + 1) as f32) * (max_v - min_v) / (num_buckets as f32) + min_v;
        }
    }

    let mut right_outliers = 0usize;
    for (i, &count) in histogram.iter().enumerate().rev() {
        right_outliers += count;
        if right_outliers > max_outliers_each_side {
            break;
        }
        if count == 0 {
            range_max = (i as f32) * (max_v - min_v) / (num_buckets as f32) + min_v;
        }
    }
    Ok((range_min, range_max))
}

fn choose_color_space_for_gainmap_math(
    base_cp: ColorPrimaries,
    alt_cp: ColorPrimaries,
) -> AvifResult<ColorPrimaries> {
    if base_cp == alt_cp {
        return Ok(base_cp);
    }
    let base_to_alt = compute_rgb_to_rgb_matrix(base_cp, alt_cp)?;
    let alt_to_base = compute_rgb_to_rgb_matrix(alt_cp, base_cp)?;
    let mut base_min = 0.0f32;
    let mut alt_min = 0.0f32;
    for c in 0..3 {
        let mut p = [0.0f32; 3];
        p[c] = 1.0;
        convert_linear_rgb(&mut p, &alt_to_base);
        base_min = base_min.min(p[0]).min(p[1]).min(p[2]);

        let mut q = [0.0f32; 3];
        q[c] = 1.0;
        convert_linear_rgb(&mut q, &base_to_alt);
        alt_min = alt_min.min(q[0]).min(q[1]).min(q[2]);
    }
    Ok(if alt_min <= base_min { base_cp } else { alt_cp })
}

pub fn compute_gainmap(
    base_image: &image::Image,
    alt_image: &image::Image,
    gainmap_width: u32,
    gainmap_height: u32,
    gainmap_depth: u8,
    gainmap_yuv_format: PixelFormat,
) -> AvifResult<GainMap> {
    if !base_image.icc.is_empty() || !alt_image.icc.is_empty() {
        return Err(AvifError::NotImplemented);
    }
    if base_image.width == 0
        || base_image.height == 0
        || !base_image.has_plane(image::Plane::Y)
        || alt_image.width == 0
        || alt_image.height == 0
        || !alt_image.has_plane(image::Plane::Y)
        || base_image.width != alt_image.width
        || base_image.height != alt_image.height
    {
        return Err(AvifError::InvalidArgument);
    }
    if !matches!(base_image.depth, 8 | 10 | 12 | 16) || !matches!(alt_image.depth, 8 | 10 | 12 | 16)
    {
        return Err(AvifError::NotImplemented);
    }
    if gainmap_width == 0 || gainmap_height == 0 || !matches!(gainmap_depth, 8 | 10 | 12) {
        return Err(AvifError::InvalidArgument);
    }

    let mut base_rgb = rgb::Image::create_from_yuv(base_image);
    base_rgb.format = if base_image.has_alpha() { rgb::Format::Rgba } else { rgb::Format::Rgb };
    base_rgb.allocate()?;
    base_rgb.convert_from_yuv(base_image)?;

    let mut alt_rgb = rgb::Image::create_from_yuv(alt_image);
    alt_rgb.format = if alt_image.has_alpha() { rgb::Format::Rgba } else { rgb::Format::Rgb };
    alt_rgb.allocate()?;
    alt_rgb.convert_from_yuv(alt_image)?;

    let width = base_image.width;
    let height = base_image.height;
    let width_usize = usize::try_from(width).map_err(|_| AvifError::OutOfMemory)?;
    let height_usize = usize::try_from(height).map_err(|_| AvifError::OutOfMemory)?;
    let num_pixels = width_usize
        .checked_mul(height_usize)
        .ok_or(AvifError::OutOfMemory)?;
    if num_pixels == 0 {
        return Err(AvifError::InvalidArgument);
    }
    let single_channel = gainmap_yuv_format == PixelFormat::Yuv400;
    let num_gm_channels = if single_channel { 1 } else { 3 };
    let mut gm_f: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for ch in &mut gm_f[..num_gm_channels] {
        *ch = try_alloc_vec(0.0f32, num_pixels)?;
    }

    let base_cp = base_image.color_primaries;
    let alt_cp = alt_image.color_primaries;
    let color_spaces_differ = base_cp != alt_cp;
    let math_cp = choose_color_space_for_gainmap_math(base_cp, alt_cp)?;
    let use_base_color_space = math_cp == base_cp;

    let base_to_linear = gamma_to_linear_fn(base_image.transfer_characteristics);
    let alt_to_linear = gamma_to_linear_fn(alt_image.transfer_characteristics);
    let y_coeffs = compute_y_coeffs(math_cp)?;

    let rgb_conv = if color_spaces_differ {
        if use_base_color_space {
            Some(compute_rgb_to_rgb_matrix(alt_cp, base_cp)?)
        } else {
            Some(compute_rgb_to_rgb_matrix(base_cp, alt_cp)?)
        }
    } else {
        None
    };

    let base_layout = RgbLayout::from_image(&base_rgb)?;
    let alt_layout = RgbLayout::from_image(&alt_rgb)?;
    let mut base_row = try_alloc_vec([0.0f32; 4], width_usize)?;
    let mut alt_row = try_alloc_vec([0.0f32; 4], width_usize)?;

    let mut base_offset = [1.0f32 / 64.0; 3];
    let mut alt_offset = [1.0f32 / 64.0; 3];

    if let Some(conv) = &rgb_conv {
        let mut channel_min = [0.0f32; 3];
        for y in 0..height {
            let src_row = if use_base_color_space {
                read_rgba_row(&alt_rgb, &alt_layout, y, &mut alt_row)?;
                &alt_row
            } else {
                read_rgba_row(&base_rgb, &base_layout, y, &mut base_row)?;
                &base_row
            };
            let to_linear = if use_base_color_space { alt_to_linear } else { base_to_linear };
            let row_offset = (y as usize) * width_usize;
            for (x, px) in src_row.iter().enumerate() {
                let mut lin = [to_linear(px[0]), to_linear(px[1]), to_linear(px[2])];
                convert_linear_rgb(&mut lin, conv);
                let idx = row_offset + x;
                if single_channel {
                    let y_lin = y_coeffs[0] * lin[0] + y_coeffs[1] * lin[1] + y_coeffs[2] * lin[2];
                    channel_min[0] = channel_min[0].min(y_lin);
                    gm_f[0][idx] = y_lin;
                } else {
                    for c in 0..3 {
                        channel_min[c] = channel_min[c].min(lin[c]);
                        gm_f[c][idx] = lin[c];
                    }
                }
            }
        }
        let max_offset = 0.1f32;
        for c in 0..num_gm_channels {
            if channel_min[c] < -EPSILON_F32 {
                if use_base_color_space {
                    alt_offset[c] = (alt_offset[c] - channel_min[c]).min(max_offset);
                } else {
                    base_offset[c] = (base_offset[c] - channel_min[c]).min(max_offset);
                }
            }
        }
    }

    let read_base = rgb_conv.is_none() || use_base_color_space;
    let read_alt = rgb_conv.is_none() || !use_base_color_space;
    let mut base_max = 1.0f32;
    let mut alt_max = 1.0f32;
    for y in 0..height {
        if read_base {
            read_rgba_row(&base_rgb, &base_layout, y, &mut base_row)?;
        }
        if read_alt {
            read_rgba_row(&alt_rgb, &alt_layout, y, &mut alt_row)?;
        }
        let row_offset = (y as usize) * width_usize;
        for x in 0..width_usize {
            let b_lin = if read_base {
                [
                    base_to_linear(base_row[x][0]),
                    base_to_linear(base_row[x][1]),
                    base_to_linear(base_row[x][2]),
                ]
            } else {
                [0.0; 3]
            };
            let a_lin = if read_alt {
                [
                    alt_to_linear(alt_row[x][0]),
                    alt_to_linear(alt_row[x][1]),
                    alt_to_linear(alt_row[x][2]),
                ]
            } else {
                [0.0; 3]
            };
            let idx = row_offset + x;
            for c in 0..num_gm_channels {
                let b_val = if !read_base {
                    gm_f[c][idx]
                } else if single_channel {
                    y_coeffs[0] * b_lin[0] + y_coeffs[1] * b_lin[1] + y_coeffs[2] * b_lin[2]
                } else {
                    b_lin[c]
                };
                let a_val = if !read_alt {
                    gm_f[c][idx]
                } else if single_channel {
                    y_coeffs[0] * a_lin[0] + y_coeffs[1] * a_lin[1] + y_coeffs[2] * a_lin[2]
                } else {
                    a_lin[c]
                };
                base_max = base_max.max(b_val);
                alt_max = alt_max.max(a_val);
                let denom = (b_val + base_offset[c]).max(EPSILON_F32);
                let ratio = (a_val + alt_offset[c]) / denom;
                gm_f[c][idx] = ratio.max(EPSILON_F32).log2();
            }
        }
    }

    let base_headroom = f64::from(base_max.max(EPSILON_F32).log2());
    let alt_headroom = f64::from(alt_max.max(EPSILON_F32).log2());
    if alt_headroom < base_headroom {
        for ch in &mut gm_f {
            for v in ch {
                *v = -*v;
            }
        }
    }

    let mut min_log2 = [0.0f32; 3];
    let mut max_log2 = [0.0f32; 3];
    for c in 0..num_gm_channels {
        let (rmin, rmax) = find_min_max_without_outliers(&gm_f[c])?;
        min_log2[c] = rmin;
        max_log2[c] = rmax;
    }

    let gamma = [UFraction(1, 1); 3];
    for c in 0..num_gm_channels {
        let range = (max_log2[c] - min_log2[c]).max(0.0);
        if range == 0.0 {
            gm_f[c].fill(0.0);
        } else {
            let gamma_f = ufraction_to_f32(gamma[c]);
            for v in &mut gm_f[c] {
                let clamped = (*v).clamp(min_log2[c], max_log2[c]);
                *v = nan_safe_clamp(((clamped - min_log2[c]) / range).powf(gamma_f));
            }
        }
    }

    let mut gm_yuv = image::Image {
        width,
        height,
        depth: gainmap_depth,
        yuv_format: gainmap_yuv_format,
        yuv_range: image::YuvRange::Full,
        matrix_coefficients: MatrixCoefficients::Bt601,
        ..Default::default()
    };
    let mut gm_rgb = rgb::Image::create_from_yuv(&gm_yuv);
    gm_rgb.format = rgb::Format::Rgb;
    gm_rgb.allocate()?;
    let gm_layout = RgbLayout::from_image(&gm_rgb)?;
    let mut gm_row = try_alloc_vec([0.0f32; 4], width_usize)?;
    for y in 0..height {
        let row_offset = (y as usize) * width_usize;
        for x in 0..width_usize {
            let r = gm_f[0][row_offset + x];
            let g = if single_channel { r } else { gm_f[1][row_offset + x] };
            let b = if single_channel { r } else { gm_f[2][row_offset + x] };
            gm_row[x] = [r, g, b, 1.0];
        }
        write_rgba_row(&mut gm_rgb, &gm_layout, y, &gm_row)?;
    }
    gm_rgb.convert_to_yuv(&mut gm_yuv)?;

    if gainmap_width != width || gainmap_height != height {
        gm_yuv.scale(gainmap_width, gainmap_height)?;
    }

    let metadata = GainMapMetadata {
        min: std::array::from_fn(|c| {
            let idx = if single_channel { 0 } else { c };
            Fraction::from(f64::from(min_log2[idx]))
        }),
        max: std::array::from_fn(|c| {
            let idx = if single_channel { 0 } else { c };
            Fraction::from(f64::from(max_log2[idx]))
        }),
        gamma,
        base_offset: std::array::from_fn(|c| {
            let idx = if single_channel { 0 } else { c };
            Fraction::from(f64::from(base_offset[idx]))
        }),
        alternate_offset: std::array::from_fn(|c| {
            let idx = if single_channel { 0 } else { c };
            Fraction::from(f64::from(alt_offset[idx]))
        }),
        base_hdr_headroom: UFraction::from(base_headroom),
        alternate_hdr_headroom: UFraction::from(alt_headroom),
        use_base_color_space,
    };
    validate_gainmap_metadata(&metadata)?;

    Ok(GainMap {
        image: gm_yuv,
        metadata,
        alt_icc: try_clone_bytes(&alt_image.icc)?,
        alt_color_primaries: alt_image.color_primaries,
        alt_transfer_characteristics: alt_image.transfer_characteristics,
        alt_matrix_coefficients: alt_image.matrix_coefficients,
        alt_yuv_range: alt_image.yuv_range,
        alt_plane_count: if alt_image.yuv_format == PixelFormat::Yuv400 { 1 } else { 3 },
        alt_plane_depth: alt_image.depth,
        alt_clli: alt_image.clli.unwrap_or_default(),
    })
}

pub fn change_base(
    image: &image::Image,
    gainmap: &GainMap,
    depth: u8,
    yuv_format: PixelFormat,
) -> AvifResult<(image::Image, GainMap)> {
    validate_gainmap_metadata(&gainmap.metadata)?;
    let headroom = ufraction_to_f32(gainmap.metadata.alternate_hdr_headroom);
    let tone_mapping_to_sdr = headroom == 0.0;

    let out_cp = if gainmap.alt_color_primaries == ColorPrimaries::Unspecified {
        image.color_primaries
    } else {
        gainmap.alt_color_primaries
    };
    let out_tc = if gainmap.alt_transfer_characteristics == TransferCharacteristics::Unspecified {
        if tone_mapping_to_sdr {
            TransferCharacteristics::Srgb
        } else {
            TransferCharacteristics::Pq
        }
    } else {
        gainmap.alt_transfer_characteristics
    };
    let mut out_mc = if gainmap.alt_matrix_coefficients == MatrixCoefficients::Unspecified {
        image.matrix_coefficients
    } else {
        gainmap.alt_matrix_coefficients
    };
    if out_mc == MatrixCoefficients::Identity && yuv_format != PixelFormat::Yuv444 {
        out_mc = MatrixCoefficients::Bt601;
    }

    let (swapped_rgb, computed_clli) =
        apply_gainmap(image, gainmap, headroom, out_cp, out_tc, depth)?;
    let clli = if !tone_mapping_to_sdr
        && gainmap.alt_clli.max_cll == 0
        && gainmap.alt_clli.max_pall == 0
    {
        Some(computed_clli)
    } else if gainmap.alt_clli.max_cll != 0 || gainmap.alt_clli.max_pall != 0 {
        Some(gainmap.alt_clli)
    } else {
        None
    };

    let has_alpha = image.has_alpha();
    let mut swapped = image::Image {
        width: image.width,
        height: image.height,
        depth,
        yuv_format,
        yuv_range: gainmap.alt_yuv_range,
        chroma_sample_position: image.chroma_sample_position,
        alpha_present: has_alpha,
        alpha_premultiplied: has_alpha && image.alpha_premultiplied,
        color_primaries: out_cp,
        transfer_characteristics: out_tc,
        matrix_coefficients: out_mc,
        clli,
        pasp: image.pasp,
        clap: image.clap,
        irot_angle: image.irot_angle,
        imir_axis: image.imir_axis,
        exif: try_clone_bytes(&image.exif)?,
        icc: Vec::new(),
        xmp: try_clone_bytes(&image.xmp)?,
        ..Default::default()
    };
    swapped_rgb.convert_to_yuv(&mut swapped)?;

    let mut swapped_meta = gainmap.metadata;
    swapped_meta.use_base_color_space = !swapped_meta.use_base_color_space;
    std::mem::swap(
        &mut swapped_meta.base_hdr_headroom,
        &mut swapped_meta.alternate_hdr_headroom,
    );
    std::mem::swap(
        &mut swapped_meta.base_offset,
        &mut swapped_meta.alternate_offset,
    );

    let swapped_gm = GainMap {
        image: gainmap.image.try_deep_clone()?,
        metadata: swapped_meta,
        alt_icc: try_clone_bytes(&image.icc)?,
        alt_color_primaries: image.color_primaries,
        alt_transfer_characteristics: image.transfer_characteristics,
        alt_matrix_coefficients: image.matrix_coefficients,
        alt_yuv_range: image.yuv_range,
        alt_plane_count: if image.yuv_format == PixelFormat::Yuv400 { 1 } else { 3 },
        alt_plane_depth: image.depth,
        alt_clli: image.clli.unwrap_or_default(),
    };
    Ok((swapped, swapped_gm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_rgb_matrix_identity_and_roundtrip() -> AvifResult<()> {
        let id = compute_rgb_to_rgb_matrix(ColorPrimaries::Srgb, ColorPrimaries::Srgb)?;
        for (i, row) in id.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < 1e-6);
            }
        }
        let fwd = compute_rgb_to_rgb_matrix(ColorPrimaries::Srgb, ColorPrimaries::Bt2020)?;
        let inv = compute_rgb_to_rgb_matrix(ColorPrimaries::Bt2020, ColorPrimaries::Srgb)?;
        let prod = mat_mul(inv, fwd);
        for (i, row) in prod.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn transfer_curve_roundtrips() {
        for tc in [
            TransferCharacteristics::Bt470m,
            TransferCharacteristics::Bt470bg,
            TransferCharacteristics::Smpte240,
            TransferCharacteristics::Linear,
            TransferCharacteristics::Log100,
            TransferCharacteristics::Log100Sqrt10,
            TransferCharacteristics::Iec61966,
            TransferCharacteristics::Bt1361,
            TransferCharacteristics::Srgb,
            TransferCharacteristics::Pq,
            TransferCharacteristics::Smpte428,
            TransferCharacteristics::Hlg,
            TransferCharacteristics::Bt709,
        ] {
            let to_lin = gamma_to_linear_fn(tc);
            let to_gam = linear_to_gamma_fn(tc);
            for sample in [0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
                let rt = to_gam(to_lin(sample));
                assert!(
                    (rt - sample).abs() < 1e-4,
                    "tc={tc:?} sample={sample} rt={rt}"
                );
            }
        }
    }

    #[test]
    fn compute_apply_and_change_base_roundtrip() -> AvifResult<()> {
        let (w, h) = (16u32, 16u32);
        let mut sdr_rgb = rgb::Image {
            width: w,
            height: h,
            depth: 8,
            format: rgb::Format::Rgb,
            ..Default::default()
        };
        sdr_rgb.allocate()?;
        for y in 0..h {
            let row = sdr_rgb.row_mut(y)?;
            for x in 0..w as usize {
                row[x * 3] = 180;
                row[x * 3 + 1] = 120;
                row[x * 3 + 2] = 60;
            }
        }
        let mut sdr = image::Image {
            width: w,
            height: h,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            yuv_range: image::YuvRange::Full,
            color_primaries: ColorPrimaries::Srgb,
            transfer_characteristics: TransferCharacteristics::Srgb,
            matrix_coefficients: MatrixCoefficients::Bt601,
            ..Default::default()
        };
        sdr_rgb.convert_to_yuv(&mut sdr)?;

        let mut hdr_rgb = rgb::Image {
            width: w,
            height: h,
            depth: 10,
            format: rgb::Format::Rgb,
            ..Default::default()
        };
        hdr_rgb.allocate()?;
        for y in 0..h {
            let row = hdr_rgb.row16_mut(y)?;
            for x in 0..w as usize {
                row[x * 3] = 750;
                row[x * 3 + 1] = 600;
                row[x * 3 + 2] = 450;
            }
        }
        let mut hdr = image::Image {
            width: w,
            height: h,
            depth: 10,
            yuv_format: PixelFormat::Yuv444,
            yuv_range: image::YuvRange::Full,
            color_primaries: ColorPrimaries::Bt2020,
            transfer_characteristics: TransferCharacteristics::Pq,
            matrix_coefficients: MatrixCoefficients::Bt2020Ncl,
            ..Default::default()
        };
        hdr_rgb.convert_to_yuv(&mut hdr)?;

        let gm = compute_gainmap(&sdr, &hdr, w, h, 8, PixelFormat::Yuv444)?;
        assert_eq!(gm.metadata.base_hdr_headroom.0, 0);
        assert!(ufraction_to_f32(gm.metadata.alternate_hdr_headroom) > 0.0);

        let (tm_rgb, clli) = apply_gainmap(
            &sdr,
            &gm,
            ufraction_to_f32(gm.metadata.alternate_hdr_headroom),
            ColorPrimaries::Bt2020,
            TransferCharacteristics::Pq,
            10,
        )?;
        assert!(clli.max_cll > 203);
        let row0 = tm_rgb.row16(0)?;
        assert!((i32::from(row0[0]) - 750).abs() <= 20, "got {}", row0[0]);

        let (swapped_img, swapped_gm) = change_base(&sdr, &gm, 10, PixelFormat::Yuv444)?;
        assert_eq!(
            swapped_img.transfer_characteristics,
            TransferCharacteristics::Pq
        );
        assert_eq!(swapped_gm.metadata.alternate_hdr_headroom.0, 0);
        assert_ne!(
            swapped_gm.metadata.use_base_color_space,
            gm.metadata.use_base_color_space
        );
        Ok(())
    }

    #[test]
    fn premultiplied_alpha_odd_dimensions_and_edge_cases() -> AvifResult<()> {
        let (w, h) = (15u32, 13u32);
        // Premultiplied RGBA input: straight RGB=(200,100,50), A=128 -> Y_prem ~= (100,50,25).
        let mut sdr_rgb = rgb::Image {
            width: w,
            height: h,
            depth: 8,
            format: rgb::Format::Rgba,
            premultiply_alpha: true,
            ..Default::default()
        };
        sdr_rgb.allocate()?;
        for y in 0..h {
            let row = sdr_rgb.row_mut(y)?;
            for x in 0..w as usize {
                row[x * 4] = 100;
                row[x * 4 + 1] = 50;
                row[x * 4 + 2] = 25;
                row[x * 4 + 3] = 128;
            }
        }
        let mut sdr = image::Image {
            width: w,
            height: h,
            depth: 8,
            yuv_format: PixelFormat::Yuv420,
            yuv_range: image::YuvRange::Full,
            alpha_present: true,
            alpha_premultiplied: true,
            color_primaries: ColorPrimaries::Srgb,
            transfer_characteristics: TransferCharacteristics::Srgb,
            matrix_coefficients: MatrixCoefficients::Bt601,
            ..Default::default()
        };
        sdr_rgb.convert_to_yuv(&mut sdr)?;

        let mut hdr_rgb = rgb::Image {
            width: w,
            height: h,
            depth: 10,
            format: rgb::Format::Rgb,
            ..Default::default()
        };
        hdr_rgb.allocate()?;
        for y in 0..h {
            let row = hdr_rgb.row16_mut(y)?;
            for x in 0..w as usize {
                row[x * 3] = 700;
                row[x * 3 + 1] = 550;
                row[x * 3 + 2] = 400;
            }
        }
        let mut hdr = image::Image {
            width: w,
            height: h,
            depth: 10,
            yuv_format: PixelFormat::Yuv420,
            yuv_range: image::YuvRange::Full,
            color_primaries: ColorPrimaries::Bt2020,
            transfer_characteristics: TransferCharacteristics::Pq,
            matrix_coefficients: MatrixCoefficients::Bt2020Ncl,
            ..Default::default()
        };
        hdr_rgb.convert_to_yuv(&mut hdr)?;

        let gm = compute_gainmap(&sdr, &hdr, w, h, 8, PixelFormat::Yuv400)?;
        assert_eq!(gm.image.matrix_coefficients, MatrixCoefficients::Bt601);
        assert_eq!(gm.metadata.min[0], gm.metadata.min[1]);
        assert_eq!(gm.metadata.min[0], gm.metadata.min[2]);
        assert_eq!(gm.metadata.base_offset[0], gm.metadata.base_offset[1]);
        assert_eq!(
            gm.metadata.alternate_offset[0],
            gm.metadata.alternate_offset[2]
        );
        validate_gainmap_metadata(&gm.metadata)?;

        // weight == 0.0 and weight != 0.0 preserve premultiplied alpha.
        let (tm_base, _) = apply_gainmap(
            &sdr,
            &gm,
            0.0,
            ColorPrimaries::Srgb,
            TransferCharacteristics::Srgb,
            8,
        )?;
        assert!(tm_base.premultiply_alpha);
        let row0 = tm_base.row(0)?;
        assert!((i32::from(row0[0]) - 100).abs() <= 4, "got r={}", row0[0]);
        assert_eq!(row0[3], 128);

        let (swapped, swapped_gm) = change_base(&sdr, &gm, 10, PixelFormat::Yuv420)?;
        assert!(swapped.alpha_premultiplied);
        let mut check_straight = rgb::Image::create_from_yuv(&swapped);
        check_straight.format = rgb::Format::Rgba;
        check_straight.allocate()?;
        check_straight.convert_from_yuv(&swapped)?;
        let got_straight = check_straight.row16(0)?[0];
        assert!(
            (i32::from(got_straight) - 700).abs() <= 40,
            "got_straight={got_straight}"
        );
        // Also check HDR-base -> SDR-alt negative weight direction.
        let (back_sdr, _) = apply_gainmap(
            &swapped,
            &swapped_gm,
            0.0,
            ColorPrimaries::Srgb,
            TransferCharacteristics::Srgb,
            8,
        )?;
        assert!(back_sdr.premultiply_alpha);
        let got_back = back_sdr.row(0)?[0];
        assert!(
            (i32::from(got_back) - 100).abs() <= 10,
            "got_back={got_back}"
        );

        // NaN and negative headroom rejection.
        assert_eq!(
            apply_gainmap(
                &sdr,
                &gm,
                f32::NAN,
                ColorPrimaries::Srgb,
                TransferCharacteristics::Srgb,
                8,
            )
            .err(),
            Some(AvifError::InvalidArgument)
        );
        assert_eq!(
            apply_gainmap(
                &sdr,
                &gm,
                -0.1,
                ColorPrimaries::Srgb,
                TransferCharacteristics::Srgb,
                8,
            )
            .err(),
            Some(AvifError::InvalidArgument)
        );

        // validate_gainmap_metadata failure cases.
        let mut bad = gm.metadata;
        bad.gamma[0] = UFraction(0, 1);
        assert_eq!(
            validate_gainmap_metadata(&bad),
            Err(AvifError::InvalidArgument)
        );
        let mut bad_range = gm.metadata;
        bad_range.min[1] = Fraction(2, 1);
        bad_range.max[1] = Fraction(1, 1);
        assert_eq!(
            validate_gainmap_metadata(&bad_range),
            Err(AvifError::InvalidArgument)
        );
        let mut bad_denom = gm.metadata;
        bad_denom.alternate_hdr_headroom = UFraction(0, 0);
        assert_eq!(
            validate_gainmap_metadata(&bad_denom),
            Err(AvifError::InvalidArgument)
        );

        // Equal base/alt headroom returns weight 0 cleanly.
        let mut equal_hm = gm.metadata;
        equal_hm.alternate_hdr_headroom = equal_hm.base_hdr_headroom;
        assert_eq!(gainmap_weight(1.0, &equal_hm), 0.0);

        // Differing non-nested gamuts (Bt470m PQ peak vs Srgb) with Yuv400 stay finite and valid.
        let mut wide_hdr = hdr.try_deep_clone()?;
        wide_hdr.color_primaries = ColorPrimaries::Bt470m;
        let gm_wide = compute_gainmap(&sdr, &wide_hdr, w, h, 8, PixelFormat::Yuv400)?;
        validate_gainmap_metadata(&gm_wide.metadata)?;
        assert_eq!(
            gm_wide.metadata.base_offset[0],
            gm_wide.metadata.base_offset[1]
        );
        Ok(())
    }

    #[test]
    fn outlier_histogram_nan_and_validation_edge_cases() -> AvifResult<()> {
        // Leading NaN followed by finite values must still be rejected.
        assert_eq!(
            find_min_max_without_outliers(&[f32::NAN, 1.0]).err(),
            Some(AvifError::InvalidArgument)
        );

        // Histogram outlier trimming on >= 2000 samples (max_outliers_each_side == 1).
        let mut samples = try_alloc_vec(0.5f32, 2000)?;
        samples[0] = -10.0;
        samples[1999] = 10.0;
        let (rmin, rmax) = find_min_max_without_outliers(&samples)?;
        assert!(rmin > -1.0 && rmin <= 0.5, "unexpected rmin={rmin}");
        assert!(rmax < 1.0 && rmax >= 0.5, "unexpected rmax={rmax}");

        // Entry validation rejects unsupported depths and missing Y planes without panicking.
        let mut img_no_planes = image::Image {
            width: 4,
            height: 4,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        let mut gm_no_planes = GainMap {
            image: img_no_planes.try_deep_clone()?,
            ..Default::default()
        };
        assert_eq!(
            apply_gainmap(
                &img_no_planes,
                &gm_no_planes,
                0.0,
                ColorPrimaries::Srgb,
                TransferCharacteristics::Srgb,
                8,
            )
            .err(),
            Some(AvifError::InvalidArgument)
        );
        assert_eq!(
            compute_gainmap(&img_no_planes, &img_no_planes, 4, 4, 8, PixelFormat::Yuv444,).err(),
            Some(AvifError::InvalidArgument)
        );

        let mut rgb = rgb::Image {
            width: 4,
            height: 4,
            depth: 8,
            format: rgb::Format::Rgb,
            ..Default::default()
        };
        rgb.allocate()?;
        rgb.convert_to_yuv(&mut img_no_planes)?;
        rgb.convert_to_yuv(&mut gm_no_planes.image)?;
        assert_eq!(
            apply_gainmap(
                &img_no_planes,
                &gm_no_planes,
                0.0,
                ColorPrimaries::Srgb,
                TransferCharacteristics::Srgb,
                0,
            )
            .err(),
            Some(AvifError::NotImplemented)
        );
        let mut bad_depth_img = img_no_planes.try_deep_clone()?;
        bad_depth_img.depth = 32;
        assert_eq!(
            compute_gainmap(&bad_depth_img, &img_no_planes, 4, 4, 8, PixelFormat::Yuv444,).err(),
            Some(AvifError::NotImplemented)
        );

        // change_base preserves gainmap.alt_yuv_range on the swapped base image.
        let mut limited_alt = img_no_planes.try_deep_clone()?;
        limited_alt.yuv_range = image::YuvRange::Limited;
        let gm = compute_gainmap(&img_no_planes, &limited_alt, 4, 4, 8, PixelFormat::Yuv444)?;
        assert_eq!(gm.alt_yuv_range, image::YuvRange::Limited);
        let (swapped, swapped_gm) = change_base(&img_no_planes, &gm, 8, PixelFormat::Yuv444)?;
        assert_eq!(swapped.yuv_range, image::YuvRange::Limited);
        assert_eq!(swapped_gm.alt_yuv_range, image::YuvRange::Full);
        Ok(())
    }
}
