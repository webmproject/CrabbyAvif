#![allow(non_snake_case)] // TODO: remove

use super::coeffs::*;
use super::rgb;
use super::rgb::*;

use crate::image;
use crate::image::Plane;
use crate::internal_utils::*;
use crate::*;

use std::cmp::min;

#[allow(unused)]
struct RgbColorSpaceInfo {
    channel_bytes: u32,
    pixel_bytes: u32,
    offset_bytes_r: usize,
    offset_bytes_g: usize,
    offset_bytes_b: usize,
    offset_bytes_a: usize,
    max_channel: i32,
    max_channel_f: f32,
}

impl RgbColorSpaceInfo {
    fn create_from(rgb: &Image) -> AvifResult<Self> {
        if !rgb.depth_valid()
            || (rgb.is_float && rgb.depth != 16)
            || (rgb.format == Format::Rgb565 && rgb.depth != 8)
        {
            return Err(AvifError::ReformatFailed);
        }
        let offsets = rgb.format.offsets();
        let max_channel = i32_from_u32((1 << rgb.depth) - 1)?;
        Ok(Self {
            channel_bytes: rgb.channel_size(),
            pixel_bytes: rgb.pixel_size(),
            offset_bytes_r: (rgb.channel_size() as usize * offsets[0]),
            offset_bytes_g: (rgb.channel_size() as usize * offsets[1]),
            offset_bytes_b: (rgb.channel_size() as usize * offsets[2]),
            offset_bytes_a: (rgb.channel_size() as usize * offsets[3]),
            max_channel,
            max_channel_f: max_channel as f32,
        })
    }
}

#[derive(PartialEq, Copy, Clone)]
enum Mode {
    YuvCoefficients(f32, f32, f32),
    Identity,
    Ycgco,
}

#[allow(unused)]
struct YuvColorSpaceInfo {
    channel_bytes: u32,
    depth: u32,
    full_range: bool,
    max_channel: u16,
    bias_y: f32,
    bias_uv: f32,
    range_y: f32,
    range_uv: f32,
    format: PixelFormat,
    mode: Mode,
}

impl YuvColorSpaceInfo {
    fn create_from(image: &image::Image) -> AvifResult<Self> {
        if !image.depth_valid() {
            return Err(AvifError::ReformatFailed);
        }
        // Unsupported matrix coefficients.
        match image.matrix_coefficients {
            MatrixCoefficients::Ycgco
            | MatrixCoefficients::Bt2020Cl
            | MatrixCoefficients::Smpte2085
            | MatrixCoefficients::ChromaDerivedCl
            | MatrixCoefficients::Ictcp => return Err(AvifError::ReformatFailed),
            _ => {}
        }
        if image.matrix_coefficients == MatrixCoefficients::Identity
            && image.yuv_format != PixelFormat::Yuv444
            && image.yuv_format != PixelFormat::Monochrome
        {
            return Err(AvifError::ReformatFailed);
        }
        let mode = match image.matrix_coefficients {
            MatrixCoefficients::Identity => Mode::Identity,
            MatrixCoefficients::Ycgco => Mode::Ycgco,
            _ => {
                let coeffs =
                    calculate_yuv_coefficients(image.color_primaries, image.matrix_coefficients);
                Mode::YuvCoefficients(coeffs[0], coeffs[1], coeffs[2])
            }
        };
        let max_channel = (1 << image.depth) - 1;
        Ok(Self {
            channel_bytes: if image.depth == 8 { 1 } else { 2 },
            depth: image.depth as u32,
            full_range: image.full_range,
            max_channel,
            bias_y: if image.full_range { 0.0 } else { (16 << (image.depth - 8)) as f32 },
            bias_uv: (1 << (image.depth - 1)) as f32,
            range_y: if image.full_range { max_channel } else { 219 << (image.depth - 8) } as f32,
            range_uv: if image.full_range { max_channel } else { 224 << (image.depth - 8) } as f32,
            format: image.yuv_format,
            mode,
        })
    }
}

struct State {
    #[allow(unused)]
    rgb: RgbColorSpaceInfo,
    #[allow(unused)]
    yuv: YuvColorSpaceInfo,
}

fn identity_yuv8_to_rgb8_full_range(image: &image::Image, rgb: &mut rgb::Image) -> AvifResult<()> {
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb565 = rgb.format == Format::Rgb565;
    let channel_count = rgb.channel_count() as usize;
    for i in 0..image.height {
        let y = image.row(Plane::Y, i)?;
        let u = image.row(Plane::U, i)?;
        let v = image.row(Plane::V, i)?;
        let rgb_pixels = rgb.row_mut(i)?;
        if rgb565 {
            unimplemented!("rgb 565 is not implemented");
        } else {
            for j in 0..image.width as usize {
                rgb_pixels[(j * channel_count) + r_offset] = v[j];
                rgb_pixels[(j * channel_count) + g_offset] = y[j];
                rgb_pixels[(j * channel_count) + b_offset] = u[j];
            }
        }
    }
    Ok(())
}

pub fn yuv_to_rgb(image: &image::Image, rgb: &mut rgb::Image) -> AvifResult<()> {
    let state = State {
        rgb: RgbColorSpaceInfo::create_from(rgb)?,
        yuv: YuvColorSpaceInfo::create_from(image)?,
    };
    if state.yuv.mode == Mode::Identity {
        if image.depth == 8
            && rgb.depth == 8
            && image.yuv_format == PixelFormat::Yuv444
            && image.full_range
        {
            return identity_yuv8_to_rgb8_full_range(image, rgb);
        }
        // TODO: Add more fast paths for identity.
        return Err(AvifError::NotImplemented);
    }
    Err(AvifError::NotImplemented)
}

fn unorm_lookup_tables(depth: u8, state: &State) -> (Vec<f32>, Vec<f32>) {
    let count = (1i32 << depth) as usize;
    let mut table_y: Vec<f32> = Vec::new();
    for cp in 0..count {
        table_y.push(((cp as f32) - state.yuv.bias_y) / state.yuv.range_y);
    }
    let mut table_uv: Vec<f32> = Vec::new();
    if state.yuv.mode == Mode::Identity {
        table_uv.extend_from_slice(&table_y[..]);
    } else {
        for cp in 0..count {
            table_uv.push(((cp as f32) - state.yuv.bias_uv) / state.yuv.range_uv);
        }
    }
    (table_y, table_uv)
}

fn compute_rgb(Y: f32, Cb: f32, Cr: f32, has_color: bool, mode: Mode) -> (f32, f32, f32) {
    let R: f32;
    let G: f32;
    let B: f32;
    if has_color {
        match mode {
            Mode::Identity => {
                G = Y;
                B = Cb;
                R = Cr;
            }
            Mode::Ycgco => {
                let t = Y - Cb;
                G = Y + Cb;
                B = t - Cr;
                R = t + Cr;
            }
            Mode::YuvCoefficients(kr, kg, kb) => {
                R = Y + (2.0 * (1.0 - kr)) * Cr;
                B = Y + (2.0 * (1.0 - kb)) * Cb;
                G = Y - ((2.0 * ((kr * (1.0 - kr) * Cr) + (kb * (1.0 - kb) * Cb))) / kg);
            }
        }
    } else {
        R = Y;
        G = Y;
        B = Y;
    }
    (
        clamp_f32(R, 0.0, 1.0),
        clamp_f32(G, 0.0, 1.0),
        clamp_f32(B, 0.0, 1.0),
    )
}

fn clamped_pixel(
    depth: u8,
    // Technically, these two are options since one of them will always be Ok.
    row: &AvifResult<&[u8]>,
    row16: &AvifResult<&[u16]>,
    index: usize,
    max_channel: u16,
) -> u16 {
    if depth == 8 {
        row.unwrap()[index] as u16
    } else {
        min(max_channel, row16.unwrap()[index])
    }
}

fn unorm_value(
    depth: u8,
    // Technically, these two are options since one of them will always be Ok.
    row: &AvifResult<&[u8]>,
    row16: &AvifResult<&[u16]>,
    index: usize,
    max_channel: u16,
    table: &[f32],
) -> f32 {
    table[clamped_pixel(depth, row, row16, index, max_channel) as usize]
}

pub fn yuv_to_rgb_any(
    image: &image::Image,
    rgb: &mut rgb::Image,
    alpha_multiply_mode: AlphaMultiplyMode,
) -> AvifResult<()> {
    let state = State {
        rgb: RgbColorSpaceInfo::create_from(rgb)?,
        yuv: YuvColorSpaceInfo::create_from(image)?,
    };
    let (table_y, table_uv) = unorm_lookup_tables(image.depth, &state);
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_depth = rgb.depth;
    let chroma_upsampling = rgb.chroma_upsampling;
    let has_color = image.has_plane(Plane::U)
        && image.has_plane(Plane::V)
        && image.yuv_format != PixelFormat::Monochrome;
    let yuv_max_channel = state.yuv.max_channel;
    let rgb_max_channel_f = state.rgb.max_channel_f;
    for j in 0..image.height as usize {
        let uv_j = j >> image.yuv_format.chroma_shift_y();
        let y_row = image.row(Plane::Y, j as u32);
        let u_row = image.row(Plane::U, uv_j as u32);
        let v_row = image.row(Plane::V, uv_j as u32);
        let a_row = image.row(Plane::A, uv_j as u32);
        let y_row16 = image.row16(Plane::Y, j as u32);
        let u_row16 = image.row16(Plane::U, uv_j as u32);
        let v_row16 = image.row16(Plane::V, uv_j as u32);
        let a_row16 = image.row16(Plane::A, j as u32);
        let (mut rgb_row, mut rgb_row16) = rgb.rows_mut(j as u32)?;
        for i in 0..image.width as usize {
            let Y = unorm_value(image.depth, &y_row, &y_row16, i, yuv_max_channel, &table_y);
            let mut Cb = 0.5;
            let mut Cr = 0.5;
            if has_color {
                let uv_i = i >> image.yuv_format.chroma_shift_x();
                if image.yuv_format == PixelFormat::Yuv444
                    || matches!(
                        chroma_upsampling,
                        ChromaUpsampling::Fastest | ChromaUpsampling::Nearest
                    )
                {
                    Cb = unorm_value(
                        image.depth,
                        &u_row,
                        &u_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                    Cr = unorm_value(
                        image.depth,
                        &v_row,
                        &v_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                } else {
                    // Bilinear filtering with weights.
                    let image_width_minus_1 = (image.width - 1) as usize;
                    let uv_adj_col: i32 = if i == 0 || (i == image_width_minus_1 && (i % 2) != 0) {
                        0
                    } else if (i % 2) != 0 {
                        1
                    } else {
                        -1
                    };
                    let u_adj_row;
                    let u_adj_row16;
                    let v_adj_row;
                    let v_adj_row16;
                    let image_height_minus_1 = (image.height - 1) as usize;
                    if j == 0
                        || (j != image_height_minus_1 && (j % 2) != 0)
                        || image.yuv_format == PixelFormat::Yuv422
                    {
                        u_adj_row = u_row;
                        u_adj_row16 = u_row16;
                        v_adj_row = v_row;
                        v_adj_row16 = v_row16;
                    } else if (j % 2) != 0 {
                        u_adj_row = image.row(Plane::U, (uv_j + 1) as u32);
                        v_adj_row = image.row(Plane::V, (uv_j + 1) as u32);
                        u_adj_row16 = image.row16(Plane::U, (uv_j + 1) as u32);
                        v_adj_row16 = image.row16(Plane::V, (uv_j + 1) as u32);
                    } else {
                        u_adj_row = image.row(Plane::U, (uv_j - 1) as u32);
                        v_adj_row = image.row(Plane::V, (uv_j - 1) as u32);
                        u_adj_row16 = image.row16(Plane::U, (uv_j - 1) as u32);
                        v_adj_row16 = image.row16(Plane::V, (uv_j - 1) as u32);
                    }
                    let mut unorm_u: [[f32; 2]; 2] = [[0.0; 2]; 2];
                    let mut unorm_v: [[f32; 2]; 2] = [[0.0; 2]; 2];
                    unorm_u[0][0] = unorm_value(
                        image.depth,
                        &u_row,
                        &u_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                    unorm_v[0][0] = unorm_value(
                        image.depth,
                        &v_row,
                        &v_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                    unorm_u[1][0] = unorm_value(
                        image.depth,
                        &u_row,
                        &u_row16,
                        ((uv_i as i32) + uv_adj_col) as usize,
                        yuv_max_channel,
                        &table_uv,
                    );
                    unorm_v[1][0] = unorm_value(
                        image.depth,
                        &v_row,
                        &v_row16,
                        ((uv_i as i32) + uv_adj_col) as usize,
                        yuv_max_channel,
                        &table_uv,
                    );
                    unorm_u[0][1] = unorm_value(
                        image.depth,
                        &u_adj_row,
                        &u_adj_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                    unorm_v[0][1] = unorm_value(
                        image.depth,
                        &v_adj_row,
                        &v_adj_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                    unorm_u[1][1] = unorm_value(
                        image.depth,
                        &u_adj_row,
                        &u_adj_row16,
                        ((uv_i as i32) + uv_adj_col) as usize,
                        yuv_max_channel,
                        &table_uv,
                    );
                    unorm_v[1][1] = unorm_value(
                        image.depth,
                        &v_adj_row,
                        &v_adj_row16,
                        ((uv_i as i32) + uv_adj_col) as usize,
                        yuv_max_channel,
                        &table_uv,
                    );
                    Cb = (unorm_u[0][0] * (9.0 / 16.0))
                        + (unorm_u[1][0] * (3.0 / 16.0))
                        + (unorm_u[0][1] * (3.0 / 16.0))
                        + (unorm_u[1][1] * (1.0 / 16.0));
                    Cr = (unorm_v[0][0] * (9.0 / 16.0))
                        + (unorm_v[1][0] * (3.0 / 16.0))
                        + (unorm_v[0][1] * (3.0 / 16.0))
                        + (unorm_v[1][1] * (1.0 / 16.0));
                }
            }
            let (mut Rc, mut Gc, mut Bc) = compute_rgb(Y, Cb, Cr, has_color, state.yuv.mode);
            if alpha_multiply_mode != AlphaMultiplyMode::NoOp {
                let unorm_a = clamped_pixel(image.depth, &a_row, &a_row16, i, yuv_max_channel);
                let Ac = clamp_f32((unorm_a as f32) / (yuv_max_channel as f32), 0.0, 1.0);
                if Ac == 0.0 {
                    Rc = 0.0;
                    Gc = 0.0;
                    Bc = 0.0;
                } else if Ac < 1.0 {
                    match alpha_multiply_mode {
                        AlphaMultiplyMode::Multiply => {
                            Rc *= Ac;
                            Gc *= Ac;
                            Bc *= Ac;
                        }
                        AlphaMultiplyMode::UnMultiply => {
                            Rc = f32::min(Rc / Ac, 1.0);
                            Gc = f32::min(Gc / Ac, 1.0);
                            Bc = f32::min(Bc / Ac, 1.0);
                        }
                        _ => {} // Not reached.
                    }
                }
            }
            if rgb_depth == 8 {
                let dst = rgb_row.as_mut().unwrap();
                dst[(i * rgb_channel_count) + r_offset] = (0.5 + (Rc * rgb_max_channel_f)) as u8;
                dst[(i * rgb_channel_count) + g_offset] = (0.5 + (Gc * rgb_max_channel_f)) as u8;
                dst[(i * rgb_channel_count) + b_offset] = (0.5 + (Bc * rgb_max_channel_f)) as u8;
            } else {
                let dst16 = rgb_row16.as_mut().unwrap();
                dst16[(i * rgb_channel_count) + r_offset] = (0.5 + (Rc * rgb_max_channel_f)) as u16;
                dst16[(i * rgb_channel_count) + g_offset] = (0.5 + (Gc * rgb_max_channel_f)) as u16;
                dst16[(i * rgb_channel_count) + b_offset] = (0.5 + (Bc * rgb_max_channel_f)) as u16;
            }
        }
    }
    Ok(())
}
