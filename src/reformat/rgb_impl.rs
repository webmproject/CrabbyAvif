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
    full_range: bool, // VideoFullRangeFlag as specified in ISO/IEC 23091-4/ITU-T H.273.
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
            // See the formulas in ISO/IEC 23091-2.
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
    rgb: RgbColorSpaceInfo,
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

fn unorm_lookup_tables(depth: u8, state: &State) -> AvifResult<(Vec<f32>, Vec<f32>)> {
    let count = 1usize << depth;
    let mut table_y: Vec<f32> = create_vec_exact(count)?;
    for cp in 0..count {
        table_y.push(((cp as f32) - state.yuv.bias_y) / state.yuv.range_y);
    }
    let mut table_uv: Vec<f32> = create_vec_exact(count)?;
    if state.yuv.mode == Mode::Identity {
        table_uv.extend_from_slice(&table_y[..]);
    } else {
        for cp in 0..count {
            table_uv.push(((cp as f32) - state.yuv.bias_uv) / state.yuv.range_uv);
        }
    }
    Ok((table_y, table_uv))
}

fn compute_rgb(y: f32, cb: f32, cr: f32, has_color: bool, mode: Mode) -> (f32, f32, f32) {
    let r: f32;
    let g: f32;
    let b: f32;
    if has_color {
        match mode {
            Mode::Identity => {
                g = y;
                b = cb;
                r = cr;
            }
            Mode::Ycgco => {
                let t = y - cb;
                g = y + cb;
                b = t - cr;
                r = t + cr;
            }
            Mode::YuvCoefficients(kr, kg, kb) => {
                r = y + (2.0 * (1.0 - kr)) * cr;
                b = y + (2.0 * (1.0 - kb)) * cb;
                g = y - ((2.0 * ((kr * (1.0 - kr) * cr) + (kb * (1.0 - kb) * cb))) / kg);
            }
        }
    } else {
        r = y;
        g = y;
        b = y;
    }
    (
        clamp_f32(r, 0.0, 1.0),
        clamp_f32(g, 0.0, 1.0),
        clamp_f32(b, 0.0, 1.0),
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
    let (table_y, table_uv) = unorm_lookup_tables(image.depth, &state)?;
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
            let y = unorm_value(image.depth, &y_row, &y_row16, i, yuv_max_channel, &table_y);
            let mut cb = 0.5;
            let mut cr = 0.5;
            if has_color {
                let uv_i = i >> image.yuv_format.chroma_shift_x();
                if image.yuv_format == PixelFormat::Yuv444
                    || matches!(
                        chroma_upsampling,
                        ChromaUpsampling::Fastest | ChromaUpsampling::Nearest
                    )
                {
                    cb = unorm_value(
                        image.depth,
                        &u_row,
                        &u_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                    cr = unorm_value(
                        image.depth,
                        &v_row,
                        &v_row16,
                        uv_i,
                        yuv_max_channel,
                        &table_uv,
                    );
                } else {
                    if image.chroma_sample_position != ChromaSamplePosition::CENTER {
                        return Err(AvifError::NotImplemented);
                    }

                    // Bilinear filtering with weights. See
                    // https://github.com/AOMediaCodec/libavif/blob/0580334466d57fedb889d5ed7ae9574d6f66e00c/src/reformat.c#L657-L685.
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
                        || (j == image_height_minus_1 && (j % 2) != 0)
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
                    cb = (unorm_u[0][0] * (9.0 / 16.0))
                        + (unorm_u[1][0] * (3.0 / 16.0))
                        + (unorm_u[0][1] * (3.0 / 16.0))
                        + (unorm_u[1][1] * (1.0 / 16.0));
                    cr = (unorm_v[0][0] * (9.0 / 16.0))
                        + (unorm_v[1][0] * (3.0 / 16.0))
                        + (unorm_v[0][1] * (3.0 / 16.0))
                        + (unorm_v[1][1] * (1.0 / 16.0));
                }
            }
            let (mut rc, mut gc, mut bc) = compute_rgb(y, cb, cr, has_color, state.yuv.mode);
            if alpha_multiply_mode != AlphaMultiplyMode::NoOp {
                let unorm_a = clamped_pixel(image.depth, &a_row, &a_row16, i, yuv_max_channel);
                let ac = clamp_f32((unorm_a as f32) / (yuv_max_channel as f32), 0.0, 1.0);
                if ac == 0.0 {
                    rc = 0.0;
                    gc = 0.0;
                    bc = 0.0;
                } else if ac < 1.0 {
                    match alpha_multiply_mode {
                        AlphaMultiplyMode::Multiply => {
                            rc *= ac;
                            gc *= ac;
                            bc *= ac;
                        }
                        AlphaMultiplyMode::UnMultiply => {
                            rc = f32::min(rc / ac, 1.0);
                            gc = f32::min(gc / ac, 1.0);
                            bc = f32::min(bc / ac, 1.0);
                        }
                        _ => {} // Not reached.
                    }
                }
            }
            if rgb_depth == 8 {
                let dst = rgb_row.as_mut().unwrap();
                dst[(i * rgb_channel_count) + r_offset] = (0.5 + (rc * rgb_max_channel_f)) as u8;
                dst[(i * rgb_channel_count) + g_offset] = (0.5 + (gc * rgb_max_channel_f)) as u8;
                dst[(i * rgb_channel_count) + b_offset] = (0.5 + (bc * rgb_max_channel_f)) as u8;
            } else {
                let dst16 = rgb_row16.as_mut().unwrap();
                dst16[(i * rgb_channel_count) + r_offset] = (0.5 + (rc * rgb_max_channel_f)) as u16;
                dst16[(i * rgb_channel_count) + g_offset] = (0.5 + (gc * rgb_max_channel_f)) as u16;
                dst16[(i * rgb_channel_count) + b_offset] = (0.5 + (bc * rgb_max_channel_f)) as u16;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuv_to_rgb() {
        fn create_420(
            matrix_coefficients: MatrixCoefficients,
            y: &[&[u8]],
            u: &[&[u8]],
            v: &[&[u8]],
        ) -> image::Image {
            let mut yuv = image::Image {
                width: y[0].len() as u32,
                height: y.len() as u32,
                depth: 8,
                yuv_format: PixelFormat::Yuv420,
                matrix_coefficients,
                ..Default::default()
            };
            assert!(yuv.allocate_planes(decoder::Category::Color).is_ok());
            for plane in image::YUV_PLANES {
                let samples = if plane == Plane::Y {
                    &y
                } else if plane == Plane::U {
                    &u
                } else {
                    &v
                };
                assert_eq!(yuv.height(plane), samples.len());
                for y in 0..yuv.height(plane) {
                    assert_eq!(yuv.width(plane), samples[y].len());
                    for x in 0..yuv.width(plane) {
                        yuv.row_mut(plane, y as u32).unwrap()[x] = samples[y][x];
                    }
                }
            }
            yuv
        }
        fn assert_near(yuv: &image::Image, r: &[&[u8]], g: &[&[u8]], b: &[&[u8]]) {
            let mut dst = rgb::Image::create_from_yuv(yuv);
            dst.format = rgb::Format::Rgb;
            dst.chroma_upsampling = ChromaUpsampling::Bilinear;
            assert!(dst.allocate().is_ok());
            assert!(yuv_to_rgb_any(yuv, &mut dst, AlphaMultiplyMode::NoOp).is_ok());
            assert_eq!(dst.height, r.len() as u32);
            assert_eq!(dst.height, g.len() as u32);
            assert_eq!(dst.height, b.len() as u32);
            for y in 0..dst.height {
                assert_eq!(dst.width, r[y as usize].len() as u32);
                assert_eq!(dst.width, g[y as usize].len() as u32);
                assert_eq!(dst.width, b[y as usize].len() as u32);
                for x in 0..dst.width {
                    let i = (x * dst.pixel_size() + 0) as usize;
                    let pixel = &dst.row(y).unwrap()[i..i + 3];
                    assert_eq!(pixel[0], r[y as usize][x as usize]);
                    assert_eq!(pixel[1], g[y as usize][x as usize]);
                    assert_eq!(pixel[2], b[y as usize][x as usize]);
                }
            }
        }

        // Testing identity 4:2:0 -> RGB would be simpler to check upsampling
        // but this is not allowed (not a real use case).
        assert_near(
            &create_420(
                MatrixCoefficients::Bt601,
                /*y=*/
                &[
                    &[0, 100, 200],  //
                    &[10, 110, 210], //
                    &[50, 150, 250],
                ],
                /*u=*/
                &[
                    &[0, 100], //
                    &[10, 110],
                ],
                /*v=*/
                &[
                    &[57, 57], //
                    &[57, 57],
                ],
            ),
            /*r=*/
            &[
                &[0, 0, 101], //
                &[0, 0, 113], //
                &[0, 43, 159],
            ],
            /*g=*/
            &[
                &[89, 196, 255],  //
                &[100, 207, 255], //
                &[145, 251, 255],
            ],
            /*b=*/
            &[
                &[0, 0, 107], //
                &[0, 0, 124], //
                &[0, 0, 181],
            ],
        );

        // Extreme values.
        assert_near(
            &create_420(
                MatrixCoefficients::Bt601,
                /*y=*/ &[&[0]],
                /*u=*/ &[&[0]],
                /*v=*/ &[&[0]],
            ),
            /*r=*/ &[&[0]],
            /*g=*/ &[&[136]],
            /*b=*/ &[&[0]],
        );
        assert_near(
            &create_420(
                MatrixCoefficients::Bt601,
                /*y=*/ &[&[255]],
                /*u=*/ &[&[255]],
                /*v=*/ &[&[255]],
            ),
            /*r=*/ &[&[255]],
            /*g=*/ &[&[125]],
            /*b=*/ &[&[255]],
        );

        // Top-right square "bleeds" into other samples during upsampling.
        assert_near(
            &create_420(
                MatrixCoefficients::Bt601,
                /*y=*/
                &[
                    &[0, 0, 255, 255],
                    &[0, 0, 255, 255],
                    &[0, 0, 0, 0],
                    &[0, 0, 0, 0],
                ],
                /*u=*/
                &[
                    &[0, 255], //
                    &[0, 0],
                ],
                /*v=*/
                &[
                    &[0, 255], //
                    &[0, 0],
                ],
            ),
            /*r=*/
            &[
                &[0, 0, 255, 255],
                &[0, 0, 255, 255],
                &[0, 0, 0, 0],
                &[0, 0, 0, 0],
            ],
            /*g=*/
            &[
                &[136, 59, 202, 125],
                &[136, 78, 255, 202],
                &[136, 116, 78, 59],
                &[136, 136, 136, 136],
            ],
            /*b=*/
            &[
                &[0, 0, 255, 255],
                &[0, 0, 255, 255],
                &[0, 0, 0, 0],
                &[0, 0, 0, 0],
            ],
        );

        // Middle square does not "bleed" into other samples during upsampling.
        assert_near(
            &create_420(
                MatrixCoefficients::Bt601,
                /*y=*/
                &[
                    &[0, 0, 0, 0],
                    &[0, 255, 255, 0],
                    &[0, 255, 255, 0],
                    &[0, 0, 0, 0],
                ],
                /*u=*/
                &[
                    &[0, 0], //
                    &[0, 0],
                ],
                /*v=*/
                &[
                    &[0, 0], //
                    &[0, 0],
                ],
            ),
            /*r=*/
            &[
                &[0, 0, 0, 0],
                &[0, 74, 74, 0],
                &[0, 74, 74, 0],
                &[0, 0, 0, 0],
            ],
            /*g=*/
            &[
                &[136, 136, 136, 136],
                &[136, 255, 255, 136],
                &[136, 255, 255, 136],
                &[136, 136, 136, 136],
            ],
            /*b=*/
            &[
                &[0, 0, 0, 0],
                &[0, 20, 20, 0],
                &[0, 20, 20, 0],
                &[0, 0, 0, 0],
            ],
        );
    }
}
