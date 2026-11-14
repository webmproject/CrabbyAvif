// Copyright 2024 Google LLC
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

use super::rgb;
use super::rgb::*;

use crate::image::Plane;
use crate::image::UninitPlane16;
use crate::image::UninitPlane8;
use crate::image::YuvRange;
use crate::internal_utils::*;
use crate::*;

use std::cmp::min;
#[cfg(target_arch = "x86_64")]
use std::mem::MaybeUninit;

// Note: truncating `as i32` matches `.floor() as i32` here because `0.5 + (v * range + bias)` is
// non-negative in-range (`trunc(x) == floor(x)` for `x >= 0`), and negative values clamp to `0`.
macro_rules! to_unorm8 {
    ($bias_y: expr, $range_y: expr, $v: expr) => {
        clamp_i32((0.5 + ($v * $range_y + $bias_y)) as i32, 0, 255) as u8
    };
}

macro_rules! to_unorm16 {
    ($bias_y: expr, $range_y: expr, $max_channel: expr, $v: expr) => {
        clamp_i32(
            (0.5 + ($v * $range_y + $bias_y)) as i32,
            0,
            $max_channel as i32,
        ) as u16
    };
}

// Copies GBR samples to YUV samples. Returns Ok(None) if not implemented.
fn identity_yuv8_to_rgb8_full_range(
    image: &image::Image,
    rgb: &mut rgb::Image,
) -> AvifResult<Option<()>> {
    if image.yuv_format != PixelFormat::Yuv444 || rgb.format == Format::Rgb565 {
        return Ok(None); // Not implemented.
    }

    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let channel_count = rgb.channel_count() as usize;
    for i in 0..image.height {
        let y = image.row(Plane::Y, i)?;
        let u = image.row(Plane::U, i)?;
        let v = image.row(Plane::V, i)?;
        let rgb_pixels = rgb.row_mut(i)?;
        for j in 0..image.width as usize {
            rgb_pixels[(j * channel_count) + r_offset] = v[j];
            rgb_pixels[(j * channel_count) + g_offset] = y[j];
            rgb_pixels[(j * channel_count) + b_offset] = u[j];
        }
    }
    Ok(Some(()))
}

// Copies GBR samples to YUV samples. Returns Ok(None) if not implemented.
fn identity_yuv16_to_rgb16_full_range(
    image: &image::Image,
    rgb: &mut rgb::Image,
) -> AvifResult<Option<()>> {
    if image.yuv_format != PixelFormat::Yuv444 {
        return Ok(None); // Not implemented.
    }

    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let channel_count = rgb.channel_count() as usize;
    for i in 0..image.height {
        let y = image.row16(Plane::Y, i)?;
        let u = image.row16(Plane::U, i)?;
        let v = image.row16(Plane::V, i)?;
        let rgb_pixels = rgb.row16_mut(i)?;
        for j in 0..image.width as usize {
            rgb_pixels[(j * channel_count) + r_offset] = v[j];
            rgb_pixels[(j * channel_count) + g_offset] = y[j];
            rgb_pixels[(j * channel_count) + b_offset] = u[j];
        }
    }
    Ok(Some(()))
}

// This is a macro and not a function because this is invoked per-pixel and there is a non-trivial
// performance impact if this is made into a function call.
macro_rules! store_rgb_pixel8 {
    ($dst:ident, $rgb_565: ident, $index: ident, $r: ident, $g: ident, $b: ident, $r_offset: ident,
     $g_offset: ident, $b_offset: ident, $rgb_channel_count: ident, $rgb_max_channel_f: ident) => {
        let r8 = (0.5 + ($r * $rgb_max_channel_f)) as u8;
        let g8 = (0.5 + ($g * $rgb_max_channel_f)) as u8;
        let b8 = (0.5 + ($b * $rgb_max_channel_f)) as u8;
        if $rgb_565 {
            // References for RGB565 color conversion:
            // * https://docs.microsoft.com/en-us/windows/win32/directshow/working-with-16-bit-rgb
            // * https://chromium.googlesource.com/libyuv/libyuv/+/9892d70c965678381d2a70a1c9002d1cf136ee78/source/row_common.cc#2362
            let r16 = ((r8 >> 3) as u16) << 11;
            let g16 = ((g8 >> 2) as u16) << 5;
            let b16 = (b8 >> 3) as u16;
            let rgb565 = (r16 | g16 | b16).to_le_bytes();
            $dst[($index * $rgb_channel_count) + $r_offset] = rgb565[0];
            $dst[($index * $rgb_channel_count) + $r_offset + 1] = rgb565[1];
        } else {
            $dst[($index * $rgb_channel_count) + $r_offset] = r8;
            $dst[($index * $rgb_channel_count) + $g_offset] = g8;
            $dst[($index * $rgb_channel_count) + $b_offset] = b8;
        }
    };
}

fn yuv8_to_rgb8_color(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let table_uv = match &table_uv {
        Some(table_uv) => table_uv,
        None => &table_y,
    };
    let table_y: &[f32; 256] = table_y
        .get(..256)
        .and_then(|s| s.try_into().ok())
        .ok_or(AvifError::ReformatFailed)?;
    let table_uv: &[f32; 256] = table_uv
        .get(..256)
        .and_then(|s| s.try_into().ok())
        .ok_or(AvifError::ReformatFailed)?;
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_565 = rgb.format == rgb::Format::Rgb565;
    let width = image.width as usize;
    if width == 0 {
        return Ok(());
    }
    let chroma_shift = image.yuv_format.chroma_shift_x();
    let uv_width = if cfg!(feature = "android_mediacodec") {
        (((width - 1) >> chroma_shift.0) << chroma_shift.1) + 1
    } else {
        ((width - 1) >> chroma_shift.0) + 1
    };
    for j in 0..image.height {
        let uv_j = j >> image.yuv_format.chroma_shift_y();
        let y_row = image
            .row(Plane::Y, j)?
            .get(..width)
            .ok_or(AvifError::ReformatFailed)?;
        let raw_u_row = image.row(Plane::U, uv_j)?;
        // If V plane is missing, then the format is NV12. In that case, set V
        // as U plane but starting at offset 1.
        let v_row = match image.row(Plane::V, uv_j) {
            Ok(row) => row,
            Err(_) => raw_u_row.get(1..).ok_or(AvifError::ReformatFailed)?,
        }
        .get(..uv_width)
        .ok_or(AvifError::ReformatFailed)?;
        let u_row = raw_u_row.get(..uv_width).ok_or(AvifError::ReformatFailed)?;
        let dst = rgb.row_mut(j)?;
        for i in 0..width {
            let uv_i = if cfg!(feature = "android_mediacodec") {
                (i >> chroma_shift.0) << chroma_shift.1
            } else {
                // chroma_shift.1 is always 0 in this case.
                i >> chroma_shift.0
            };
            let y = table_y[y_row[i] as usize];
            let cb = table_uv[u_row[uv_i] as usize];
            let cr = table_uv[v_row[uv_i] as usize];
            let r = y + (2.0 * (1.0 - kr)) * cr;
            let b = y + (2.0 * (1.0 - kb)) * cb;
            let g = y - ((2.0 * ((kr * (1.0 - kr) * cr) + (kb * (1.0 - kb) * cb))) / kg);
            let r = clamp_f32(r, 0.0, 1.0);
            let g = clamp_f32(g, 0.0, 1.0);
            let b = clamp_f32(b, 0.0, 1.0);
            store_rgb_pixel8!(
                dst,
                rgb_565,
                i,
                r,
                g,
                b,
                r_offset,
                g_offset,
                b_offset,
                rgb_channel_count,
                rgb_max_channel_f
            );
        }
    }
    Ok(())
}

fn yuv16_to_rgb16_color(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let table_uv = match &table_uv {
        Some(table_uv) => table_uv,
        None => &table_y,
    };
    let yuv_max_channel = image.max_channel();
    let mask = yuv_max_channel as usize;
    let table_y = table_y.get(..=mask).ok_or(AvifError::ReformatFailed)?;
    let table_uv = table_uv.get(..=mask).ok_or(AvifError::ReformatFailed)?;
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    if r_offset >= rgb_channel_count
        || g_offset >= rgb_channel_count
        || b_offset >= rgb_channel_count
    {
        return AvifError::reformat_failed();
    }
    let width = image.width as usize;
    if width == 0 {
        return Ok(());
    }
    let chroma_shift = image.yuv_format.chroma_shift_x();
    let uv_width = if cfg!(feature = "android_mediacodec") {
        (((width - 1) >> chroma_shift.0) << chroma_shift.1) + 1
    } else {
        ((width - 1) >> chroma_shift.0) + 1
    };
    for j in 0..image.height {
        let uv_j = j >> image.yuv_format.chroma_shift_y();
        let y_row = image
            .row16(Plane::Y, j)?
            .get(..width)
            .ok_or(AvifError::ReformatFailed)?;
        let raw_u_row = image.row16(Plane::U, uv_j)?;
        // If V plane is missing, then the format is P010. In that case, set V
        // as U plane but starting at offset 1.
        let v_row = match image.row16(Plane::V, uv_j) {
            Ok(row) => row,
            Err(_) => raw_u_row.get(1..).ok_or(AvifError::ReformatFailed)?,
        }
        .get(..uv_width)
        .ok_or(AvifError::ReformatFailed)?;
        let u_row = raw_u_row.get(..uv_width).ok_or(AvifError::ReformatFailed)?;
        let dst = rgb
            .row16_mut(j)?
            .get_mut(..width * rgb_channel_count)
            .ok_or(AvifError::ReformatFailed)?;
        for (i, (&y_sample, dst_pixel)) in y_row
            .iter()
            .zip(dst.chunks_exact_mut(rgb_channel_count))
            .enumerate()
        {
            let uv_i = if cfg!(feature = "android_mediacodec") {
                (i >> chroma_shift.0) << chroma_shift.1
            } else {
                // chroma_shift.1 is always 0 in this case.
                i >> chroma_shift.0
            };
            let y = table_y[min(y_sample, yuv_max_channel) as usize & mask];
            let cb = table_uv[min(u_row[uv_i], yuv_max_channel) as usize & mask];
            let cr = table_uv[min(v_row[uv_i], yuv_max_channel) as usize & mask];
            let r = y + (2.0 * (1.0 - kr)) * cr;
            let b = y + (2.0 * (1.0 - kb)) * cb;
            let g = y - ((2.0 * ((kr * (1.0 - kr) * cr) + (kb * (1.0 - kb) * cb))) / kg);
            let r = clamp_f32(r, 0.0, 1.0);
            let g = clamp_f32(g, 0.0, 1.0);
            let b = clamp_f32(b, 0.0, 1.0);
            dst_pixel[r_offset] = (0.5 + (r * rgb_max_channel_f)) as u16;
            dst_pixel[g_offset] = (0.5 + (g * rgb_max_channel_f)) as u16;
            dst_pixel[b_offset] = (0.5 + (b * rgb_max_channel_f)) as u16;
        }
    }
    Ok(())
}

fn yuv16_to_rgb8_color(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let table_uv = match &table_uv {
        Some(table_uv) => table_uv,
        None => &table_y,
    };
    let yuv_max_channel = image.max_channel();
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_565 = rgb.format == rgb::Format::Rgb565;
    let chroma_shift = image.yuv_format.chroma_shift_x();
    for j in 0..image.height {
        let uv_j = j >> image.yuv_format.chroma_shift_y();
        let y_row = image.row16(Plane::Y, j)?;
        let u_row = image.row16(Plane::U, uv_j)?;
        // If V plane is missing, then the format is P010. In that case, set V
        // as U plane but starting at offset 1.
        let v_row = image.row16(Plane::V, uv_j).unwrap_or(&u_row[1..]);
        let dst = rgb.row_mut(j)?;
        for i in 0..image.width as usize {
            let uv_i = if cfg!(feature = "android_mediacodec") {
                (i >> chroma_shift.0) << chroma_shift.1
            } else {
                // chroma_shift.1 is always 0 in this case.
                i >> chroma_shift.0
            };
            let y = table_y[min(y_row[i], yuv_max_channel) as usize];
            let cb = table_uv[min(u_row[uv_i], yuv_max_channel) as usize];
            let cr = table_uv[min(v_row[uv_i], yuv_max_channel) as usize];
            let r = y + (2.0 * (1.0 - kr)) * cr;
            let b = y + (2.0 * (1.0 - kb)) * cb;
            let g = y - ((2.0 * ((kr * (1.0 - kr) * cr) + (kb * (1.0 - kb) * cb))) / kg);
            let r = clamp_f32(r, 0.0, 1.0);
            let g = clamp_f32(g, 0.0, 1.0);
            let b = clamp_f32(b, 0.0, 1.0);
            store_rgb_pixel8!(
                dst,
                rgb_565,
                i,
                r,
                g,
                b,
                r_offset,
                g_offset,
                b_offset,
                rgb_channel_count,
                rgb_max_channel_f
            );
        }
    }
    Ok(())
}

fn yuv8_to_rgb16_color(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let table_uv = match &table_uv {
        Some(table_uv) => table_uv,
        None => &table_y,
    };
    let table_y: &[f32; 256] = table_y
        .get(..256)
        .and_then(|s| s.try_into().ok())
        .ok_or(AvifError::ReformatFailed)?;
    let table_uv: &[f32; 256] = table_uv
        .get(..256)
        .and_then(|s| s.try_into().ok())
        .ok_or(AvifError::ReformatFailed)?;
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    if r_offset >= rgb_channel_count
        || g_offset >= rgb_channel_count
        || b_offset >= rgb_channel_count
    {
        return AvifError::reformat_failed();
    }
    let width = image.width as usize;
    if width == 0 {
        return Ok(());
    }
    let chroma_shift = image.yuv_format.chroma_shift_x();
    let uv_width = if cfg!(feature = "android_mediacodec") {
        (((width - 1) >> chroma_shift.0) << chroma_shift.1) + 1
    } else {
        ((width - 1) >> chroma_shift.0) + 1
    };
    for j in 0..image.height {
        let uv_j = j >> image.yuv_format.chroma_shift_y();
        let y_row = image
            .row(Plane::Y, j)?
            .get(..width)
            .ok_or(AvifError::ReformatFailed)?;
        let raw_u_row = image.row(Plane::U, uv_j)?;
        // If V plane is missing, then the format is NV12. In that case, set V
        // as U plane but starting at offset 1.
        let v_row = match image.row(Plane::V, uv_j) {
            Ok(row) => row,
            Err(_) => raw_u_row.get(1..).ok_or(AvifError::ReformatFailed)?,
        }
        .get(..uv_width)
        .ok_or(AvifError::ReformatFailed)?;
        let u_row = raw_u_row.get(..uv_width).ok_or(AvifError::ReformatFailed)?;
        let dst = rgb
            .row16_mut(j)?
            .get_mut(..width * rgb_channel_count)
            .ok_or(AvifError::ReformatFailed)?;
        for (i, (&y_sample, dst_pixel)) in y_row
            .iter()
            .zip(dst.chunks_exact_mut(rgb_channel_count))
            .enumerate()
        {
            let uv_i = if cfg!(feature = "android_mediacodec") {
                (i >> chroma_shift.0) << chroma_shift.1
            } else {
                // chroma_shift.1 is always 0 in this case.
                i >> chroma_shift.0
            };
            let y = table_y[y_sample as usize];
            let cb = table_uv[u_row[uv_i] as usize];
            let cr = table_uv[v_row[uv_i] as usize];
            let r = y + (2.0 * (1.0 - kr)) * cr;
            let b = y + (2.0 * (1.0 - kb)) * cb;
            let g = y - ((2.0 * ((kr * (1.0 - kr) * cr) + (kb * (1.0 - kb) * cb))) / kg);
            let r = clamp_f32(r, 0.0, 1.0);
            let g = clamp_f32(g, 0.0, 1.0);
            let b = clamp_f32(b, 0.0, 1.0);
            dst_pixel[r_offset] = (0.5 + (r * rgb_max_channel_f)) as u16;
            dst_pixel[g_offset] = (0.5 + (g * rgb_max_channel_f)) as u16;
            dst_pixel[b_offset] = (0.5 + (b * rgb_max_channel_f)) as u16;
        }
    }
    Ok(())
}

fn yuv8_to_rgb8_monochrome(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, _table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_565 = rgb.format == rgb::Format::Rgb565;
    for j in 0..image.height {
        let y_row = image.row(Plane::Y, j)?;
        let dst = rgb.row_mut(j)?;
        for i in 0..image.width as usize {
            let y = table_y[y_row[i] as usize];
            store_rgb_pixel8!(
                dst,
                rgb_565,
                i,
                y,
                y,
                y,
                r_offset,
                g_offset,
                b_offset,
                rgb_channel_count,
                rgb_max_channel_f
            );
        }
    }
    Ok(())
}

fn yuv16_to_rgb16_monochrome(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, _table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let yuv_max_channel = image.max_channel();
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    for j in 0..image.height {
        let y_row = image.row16(Plane::Y, j)?;
        let dst = rgb.row16_mut(j)?;
        for i in 0..image.width as usize {
            let y = table_y[min(y_row[i], yuv_max_channel) as usize];
            let rgb_pixel = (0.5 + (y * rgb_max_channel_f)) as u16;
            dst[(i * rgb_channel_count) + r_offset] = rgb_pixel;
            dst[(i * rgb_channel_count) + g_offset] = rgb_pixel;
            dst[(i * rgb_channel_count) + b_offset] = rgb_pixel;
        }
    }
    Ok(())
}

fn yuv16_to_rgb8_monochrome(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, _table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let yuv_max_channel = image.max_channel();
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_565 = rgb.format == rgb::Format::Rgb565;
    for j in 0..image.height {
        let y_row = image.row16(Plane::Y, j)?;
        let dst = rgb.row_mut(j)?;
        for i in 0..image.width as usize {
            let y = table_y[min(y_row[i], yuv_max_channel) as usize];
            store_rgb_pixel8!(
                dst,
                rgb_565,
                i,
                y,
                y,
                y,
                r_offset,
                g_offset,
                b_offset,
                rgb_channel_count,
                rgb_max_channel_f
            );
        }
    }
    Ok(())
}

fn yuv8_to_rgb16_monochrome(
    image: &image::Image,
    rgb: &mut rgb::Image,
    kr: f32,
    kg: f32,
    kb: f32,
) -> AvifResult<()> {
    let (table_y, _table_uv) = unorm_lookup_tables(image, Mode::YuvCoefficients(kr, kg, kb))?;
    let rgb_max_channel_f = rgb.max_channel_f();
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    for j in 0..image.height {
        let y_row = image.row(Plane::Y, j)?;
        let dst = rgb.row16_mut(j)?;
        for i in 0..image.width as usize {
            let y = table_y[y_row[i] as usize];
            let rgb_pixel = (0.5 + (y * rgb_max_channel_f)) as u16;
            dst[(i * rgb_channel_count) + r_offset] = rgb_pixel;
            dst[(i * rgb_channel_count) + g_offset] = rgb_pixel;
            dst[(i * rgb_channel_count) + b_offset] = rgb_pixel;
        }
    }
    Ok(())
}

// Converts RGB samples to YUV samples. Returns Ok(None) if not implemented.
pub(crate) fn yuv_to_rgb_fast(
    image: &image::Image,
    rgb: &mut rgb::Image,
) -> AvifResult<Option<()>> {
    let mode: Mode = image.into();
    Ok(match mode {
        Mode::Identity => match (image.depth, rgb.depth, image.yuv_range) {
            (8, 8, YuvRange::Full) => identity_yuv8_to_rgb8_full_range(image, rgb)?,
            (16, 16, YuvRange::Full) => identity_yuv16_to_rgb16_full_range(image, rgb)?,
            _ => None,
        },
        Mode::YuvCoefficients(kr, kg, kb) => {
            let has_color = image.yuv_format != PixelFormat::Yuv400;
            if !cfg!(feature = "android_mediacodec") {
                // In this case, P010 and NV12 formats are not supported.
                assert_eq!(image.yuv_format.chroma_shift_x().1, 0);
            }
            Some(match (image.depth == 8, rgb.depth == 8, has_color) {
                (true, true, true) => yuv8_to_rgb8_color(image, rgb, kr, kg, kb),
                (false, false, true) => yuv16_to_rgb16_color(image, rgb, kr, kg, kb),
                (false, true, true) => yuv16_to_rgb8_color(image, rgb, kr, kg, kb),
                (true, false, true) => yuv8_to_rgb16_color(image, rgb, kr, kg, kb),
                (true, true, false) => yuv8_to_rgb8_monochrome(image, rgb, kr, kg, kb),
                (false, false, false) => yuv16_to_rgb16_monochrome(image, rgb, kr, kg, kb),
                (false, true, false) => yuv16_to_rgb8_monochrome(image, rgb, kr, kg, kb),
                (true, false, false) => yuv8_to_rgb16_monochrome(image, rgb, kr, kg, kb),
            }?)
        }
        Mode::Ycgco | Mode::YcgcoRe | Mode::YcgcoRo => None, // Not implemented
    })
}

fn bias_and_range_y(image: &image::Image) -> (f32, f32) {
    // Formula specified in ISO/IEC 23091-2.
    if image.yuv_range == YuvRange::Limited {
        (
            (16 << (image.depth - 8)) as f32,
            (219 << (image.depth - 8)) as f32,
        )
    } else {
        (0.0, image.max_channel_f())
    }
}

fn bias_and_range_uv(image: &image::Image) -> (f32, f32) {
    // Formula specified in ISO/IEC 23091-2.
    (
        (1 << (image.depth - 1)) as f32,
        if image.yuv_range == YuvRange::Limited {
            (224 << (image.depth - 8)) as f32
        } else {
            image.max_channel_f()
        },
    )
}

fn unorm_lookup_tables(
    image: &image::Image,
    mode: Mode,
) -> AvifResult<(Vec<f32>, Option<Vec<f32>>)> {
    let count = 1usize << image.depth;
    let mut table_y: Vec<f32> = create_vec_exact(count)?;
    let (bias_y, range_y) = bias_and_range_y(image);
    for cp in 0..count {
        table_y.try_push(((cp as f32) - bias_y) / range_y)?;
    }
    if mode == Mode::Identity {
        Ok((table_y, None))
    } else {
        let (bias_uv, range_uv) = bias_and_range_uv(image);
        let mut table_uv: Vec<f32> = create_vec_exact(count)?;
        for cp in 0..count {
            table_uv.try_push(((cp as f32) - bias_uv) / range_uv)?;
        }
        Ok((table_y, Some(table_uv)))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_rgb(
    y: f32,
    cb: f32,
    cr: f32,
    has_color: bool,
    mode: Mode,
    clamped_y: u16,
    yuv_max_channel: u16,
    rgb_max_channel: u16,
    rgb_max_channel_f: f32,
) -> (f32, f32, f32) {
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
            Mode::YcgcoRe | Mode::YcgcoRo => {
                // Equations (62) through (65) in https://www.itu.int/rec/T-REC-H.273
                let cg = (0.5 + cb * yuv_max_channel as f32).floor() as i32;
                let co = (0.5 + cr * yuv_max_channel as f32).floor() as i32;
                let t = clamped_y as i32 - (cg >> 1);
                let rgb_max_channel = rgb_max_channel as i32;
                g = clamp_i32(t + cg, 0, rgb_max_channel) as f32 / rgb_max_channel_f;
                let tmp_b = clamp_i32(t - (co >> 1), 0, rgb_max_channel) as f32;
                b = tmp_b / rgb_max_channel_f;
                r = clamp_i32(tmp_b as i32 + co, 0, rgb_max_channel) as f32 / rgb_max_channel_f;
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

fn yuv16_to_rgb_any(
    image: &image::Image,
    rgb: &mut rgb::Image,
    alpha_multiply_mode: AlphaMultiplyMode,
    fast_or_no_chroma_subsampling: bool,
) -> AvifResult<()> {
    if image.width == 0 || image.height == 0 {
        return Ok(());
    }
    let mode: Mode = image.into();
    let (table_y, table_uv) = unorm_lookup_tables(image, mode)?;
    let table_uv = match &table_uv {
        Some(table_uv) => table_uv,
        None => &table_y,
    };
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let yuv_has_color = image.has_plane(Plane::U)
        && image.has_plane(Plane::V)
        && image.yuv_format != PixelFormat::Yuv400;
    let rgb_has_color = !rgb.format.is_gray();
    if rgb_channel_count == 0
        || r_offset >= rgb_channel_count
        || (rgb_has_color && (g_offset >= rgb_channel_count || b_offset >= rgb_channel_count))
    {
        return AvifError::reformat_failed();
    }
    let yuv_max_channel = image.max_channel();
    let mask = yuv_max_channel as usize;
    let table_y = table_y.get(..=mask).ok_or(AvifError::ReformatFailed)?;
    let table_uv = table_uv.get(..=mask).ok_or(AvifError::ReformatFailed)?;
    let rgb_max_channel = rgb.max_channel();
    let rgb_max_channel_f = rgb.max_channel_f();
    let chroma_shift = image.yuv_format.chroma_shift_x();
    let image_width = image.width as usize;
    let image_width_minus_1 = image_width.saturating_sub(1);
    let uv_width = if image_width > 0 {
        ((image_width_minus_1 >> chroma_shift.0) << chroma_shift.1) + 1
    } else {
        0
    };
    for j in 0..image.height {
        let uv_j = j >> image.yuv_format.chroma_shift_y();
        let y_row = image
            .row16(Plane::Y, j)?
            .get(..image_width)
            .ok_or(AvifError::ReformatFailed)?;
        let (u_row, v_row, u_adj_row, v_adj_row): (&[u16], &[u16], &[u16], &[u16]) =
            if yuv_has_color {
                let u_row = image
                    .row16(Plane::U, uv_j)?
                    .get(..uv_width)
                    .ok_or(AvifError::ReformatFailed)?;
                let v_row = image
                    .row16(Plane::V, uv_j)?
                    .get(..uv_width)
                    .ok_or(AvifError::ReformatFailed)?;
                if fast_or_no_chroma_subsampling {
                    (u_row, v_row, u_row, v_row)
                } else {
                    let uv_adj_j = if j == 0
                        || (j == image.height - 1 && (j % 2) != 0)
                        || image.yuv_format == PixelFormat::Yuv422
                    {
                        uv_j
                    } else if (j % 2) != 0 {
                        uv_j + 1
                    } else {
                        uv_j - 1
                    };
                    (
                        u_row,
                        v_row,
                        image
                            .row16(Plane::U, uv_adj_j)?
                            .get(..uv_width)
                            .ok_or(AvifError::ReformatFailed)?,
                        image
                            .row16(Plane::V, uv_adj_j)?
                            .get(..uv_width)
                            .ok_or(AvifError::ReformatFailed)?,
                    )
                }
            } else {
                (&[], &[], &[], &[])
            };
        let a_row: &[u16] = if alpha_multiply_mode != AlphaMultiplyMode::NoOp {
            image
                .row16(Plane::A, j)?
                .get(..image_width)
                .ok_or(AvifError::ReformatFailed)?
        } else {
            &[]
        };
        macro_rules! convert_row {
            ($dst_row:expr, $as_ty:ty) => {{
                let dst = $dst_row
                    .get_mut(..image_width * rgb_channel_count)
                    .ok_or(AvifError::ReformatFailed)?;
                for (i, (&y_sample, dst_pixel)) in y_row
                    .iter()
                    .zip(dst.chunks_exact_mut(rgb_channel_count))
                    .enumerate()
                {
                    let clamped_y = min(y_sample, yuv_max_channel);
                    let y = table_y[min(y_sample as usize, mask)];
                    let mut cb = 0.5;
                    let mut cr = 0.5;
                    if yuv_has_color {
                        let uv_i = (i >> chroma_shift.0) << chroma_shift.1;
                        if fast_or_no_chroma_subsampling {
                            cb = table_uv[min(u_row[uv_i] as usize, mask)];
                            cr = table_uv[min(v_row[uv_i] as usize, mask)];
                        } else {
                            // Bilinear filtering with weights. See
                            // https://github.com/AOMediaCodec/libavif/blob/0580334466d57fedb889d5ed7ae9574d6f66e00c/src/reformat.c#L657-L685.
                            let uv_adj_i = if i == 0 || (i == image_width_minus_1 && (i % 2) != 0) {
                                uv_i
                            } else if (i % 2) != 0 {
                                uv_i + 1
                            } else {
                                uv_i - 1
                            };

                            let unorm_u = [
                                table_uv[min(u_row[uv_i] as usize, mask)],
                                table_uv[min(u_row[uv_adj_i] as usize, mask)],
                                table_uv[min(u_adj_row[uv_i] as usize, mask)],
                                table_uv[min(u_adj_row[uv_adj_i] as usize, mask)],
                            ];
                            cb = (unorm_u[0] * (9.0 / 16.0))
                                + (unorm_u[1] * (3.0 / 16.0))
                                + (unorm_u[2] * (3.0 / 16.0))
                                + (unorm_u[3] * (1.0 / 16.0));

                            let unorm_v = [
                                table_uv[min(v_row[uv_i] as usize, mask)],
                                table_uv[min(v_row[uv_adj_i] as usize, mask)],
                                table_uv[min(v_adj_row[uv_i] as usize, mask)],
                                table_uv[min(v_adj_row[uv_adj_i] as usize, mask)],
                            ];
                            cr = (unorm_v[0] * (9.0 / 16.0))
                                + (unorm_v[1] * (3.0 / 16.0))
                                + (unorm_v[2] * (3.0 / 16.0))
                                + (unorm_v[3] * (1.0 / 16.0));
                        }
                    }
                    let (mut rc, mut gc, mut bc) = if rgb_has_color {
                        compute_rgb(
                            y,
                            cb,
                            cr,
                            yuv_has_color,
                            mode,
                            clamped_y,
                            yuv_max_channel,
                            rgb_max_channel,
                            rgb_max_channel_f,
                        )
                    } else {
                        (clamp_f32(y, 0.0, 1.0), 0.0, 0.0)
                    };
                    if alpha_multiply_mode != AlphaMultiplyMode::NoOp {
                        let unorm_a = min(a_row[i], yuv_max_channel);
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
                    dst_pixel[r_offset] = (0.5 + (rc * rgb_max_channel_f)) as $as_ty;
                    if rgb_has_color {
                        dst_pixel[g_offset] = (0.5 + (gc * rgb_max_channel_f)) as $as_ty;
                        dst_pixel[b_offset] = (0.5 + (bc * rgb_max_channel_f)) as $as_ty;
                    }
                }
            }};
        }
        if rgb.depth == 8 {
            convert_row!(rgb.row_mut(j)?, u8);
        } else {
            convert_row!(rgb.row16_mut(j)?, u16);
        }
    }
    Ok(())
}

fn yuv8_to_rgb_any(
    image: &image::Image,
    rgb: &mut rgb::Image,
    alpha_multiply_mode: AlphaMultiplyMode,
    fast_or_no_chroma_subsampling: bool,
) -> AvifResult<()> {
    if image.width == 0 || image.height == 0 {
        return Ok(());
    }
    let mode: Mode = image.into();
    let (table_y, table_uv) = unorm_lookup_tables(image, mode)?;
    let table_uv = match &table_uv {
        Some(table_uv) => table_uv,
        None => &table_y,
    };
    let table_y: &[f32; 256] = table_y
        .get(..256)
        .and_then(|s| s.try_into().ok())
        .ok_or(AvifError::ReformatFailed)?;
    let table_uv: &[f32; 256] = table_uv
        .get(..256)
        .and_then(|s| s.try_into().ok())
        .ok_or(AvifError::ReformatFailed)?;
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let yuv_has_color = image.has_plane(Plane::U)
        && image.has_plane(Plane::V)
        && image.yuv_format != PixelFormat::Yuv400;
    let rgb_has_color = !rgb.format.is_gray();
    if rgb_channel_count == 0
        || r_offset >= rgb_channel_count
        || (rgb_has_color && (g_offset >= rgb_channel_count || b_offset >= rgb_channel_count))
    {
        return AvifError::reformat_failed();
    }
    let yuv_max_channel = image.max_channel();
    let rgb_max_channel = rgb.max_channel();
    let rgb_max_channel_f = rgb.max_channel_f();
    let chroma_shift = image.yuv_format.chroma_shift_x();
    let image_width = image.width as usize;
    let image_width_minus_1 = image_width.saturating_sub(1);
    let uv_width = if image_width > 0 {
        ((image_width_minus_1 >> chroma_shift.0) << chroma_shift.1) + 1
    } else {
        0
    };
    for j in 0..image.height {
        let uv_j = j >> image.yuv_format.chroma_shift_y();
        let y_row = image
            .row(Plane::Y, j)?
            .get(..image_width)
            .ok_or(AvifError::ReformatFailed)?;
        let (u_row, v_row, u_adj_row, v_adj_row): (&[u8], &[u8], &[u8], &[u8]) = if yuv_has_color {
            let u_row = image
                .row(Plane::U, uv_j)?
                .get(..uv_width)
                .ok_or(AvifError::ReformatFailed)?;
            let v_row = image
                .row(Plane::V, uv_j)?
                .get(..uv_width)
                .ok_or(AvifError::ReformatFailed)?;
            if fast_or_no_chroma_subsampling {
                (u_row, v_row, u_row, v_row)
            } else {
                let uv_adj_j = if j == 0
                    || (j == image.height - 1 && (j % 2) != 0)
                    || image.yuv_format == PixelFormat::Yuv422
                {
                    uv_j
                } else if (j % 2) != 0 {
                    uv_j + 1
                } else {
                    uv_j - 1
                };
                (
                    u_row,
                    v_row,
                    image
                        .row(Plane::U, uv_adj_j)?
                        .get(..uv_width)
                        .ok_or(AvifError::ReformatFailed)?,
                    image
                        .row(Plane::V, uv_adj_j)?
                        .get(..uv_width)
                        .ok_or(AvifError::ReformatFailed)?,
                )
            }
        } else {
            (&[], &[], &[], &[])
        };
        let a_row: &[u8] = if alpha_multiply_mode != AlphaMultiplyMode::NoOp {
            image
                .row(Plane::A, j)?
                .get(..image_width)
                .ok_or(AvifError::ReformatFailed)?
        } else {
            &[]
        };
        macro_rules! convert_row {
            ($dst_row:expr, $as_ty:ty) => {{
                let dst = $dst_row
                    .get_mut(..image_width * rgb_channel_count)
                    .ok_or(AvifError::ReformatFailed)?;
                for (i, (&y_sample, dst_pixel)) in y_row
                    .iter()
                    .zip(dst.chunks_exact_mut(rgb_channel_count))
                    .enumerate()
                {
                    let clamped_y = y_sample as u16;
                    let y = table_y[y_sample as usize];
                    let mut cb = 0.5;
                    let mut cr = 0.5;
                    if yuv_has_color {
                        let uv_i = (i >> chroma_shift.0) << chroma_shift.1;
                        if fast_or_no_chroma_subsampling {
                            cb = table_uv[u_row[uv_i] as usize];
                            cr = table_uv[v_row[uv_i] as usize];
                        } else {
                            // Bilinear filtering with weights. See
                            // https://github.com/AOMediaCodec/libavif/blob/0580334466d57fedb889d5ed7ae9574d6f66e00c/src/reformat.c#L657-L685.
                            let uv_adj_i = if i == 0 || (i == image_width_minus_1 && (i % 2) != 0) {
                                uv_i
                            } else if (i % 2) != 0 {
                                uv_i + 1
                            } else {
                                uv_i - 1
                            };

                            let unorm_u = [
                                table_uv[u_row[uv_i] as usize],
                                table_uv[u_row[uv_adj_i] as usize],
                                table_uv[u_adj_row[uv_i] as usize],
                                table_uv[u_adj_row[uv_adj_i] as usize],
                            ];
                            cb = (unorm_u[0] * (9.0 / 16.0))
                                + (unorm_u[1] * (3.0 / 16.0))
                                + (unorm_u[2] * (3.0 / 16.0))
                                + (unorm_u[3] * (1.0 / 16.0));

                            let unorm_v = [
                                table_uv[v_row[uv_i] as usize],
                                table_uv[v_row[uv_adj_i] as usize],
                                table_uv[v_adj_row[uv_i] as usize],
                                table_uv[v_adj_row[uv_adj_i] as usize],
                            ];
                            cr = (unorm_v[0] * (9.0 / 16.0))
                                + (unorm_v[1] * (3.0 / 16.0))
                                + (unorm_v[2] * (3.0 / 16.0))
                                + (unorm_v[3] * (1.0 / 16.0));
                        }
                    }
                    let (mut rc, mut gc, mut bc) = if rgb_has_color {
                        compute_rgb(
                            y,
                            cb,
                            cr,
                            yuv_has_color,
                            mode,
                            clamped_y,
                            yuv_max_channel,
                            rgb_max_channel,
                            rgb_max_channel_f,
                        )
                    } else {
                        (clamp_f32(y, 0.0, 1.0), 0.0, 0.0)
                    };
                    if alpha_multiply_mode != AlphaMultiplyMode::NoOp {
                        let unorm_a = a_row[i];
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
                    dst_pixel[r_offset] = (0.5 + (rc * rgb_max_channel_f)) as $as_ty;
                    if rgb_has_color {
                        dst_pixel[g_offset] = (0.5 + (gc * rgb_max_channel_f)) as $as_ty;
                        dst_pixel[b_offset] = (0.5 + (bc * rgb_max_channel_f)) as $as_ty;
                    }
                }
            }};
        }
        if rgb.depth == 8 {
            convert_row!(rgb.row_mut(j)?, u8);
        } else {
            convert_row!(rgb.row16_mut(j)?, u16);
        }
    }
    Ok(())
}

pub(crate) fn yuv_to_rgb_any(
    image: &image::Image,
    rgb: &mut rgb::Image,
    alpha_multiply_mode: AlphaMultiplyMode,
) -> AvifResult<()> {
    let fast_or_no_chroma_subsampling = image.yuv_format == PixelFormat::Yuv444
        || matches!(
            rgb.chroma_upsampling,
            ChromaUpsampling::Fastest | ChromaUpsampling::Nearest
        );
    if !fast_or_no_chroma_subsampling
        && image.chroma_sample_position != ChromaSamplePosition::CENTER
    {
        return AvifError::not_implemented();
    }
    if image.depth > 8 {
        yuv16_to_rgb_any(
            image,
            rgb,
            alpha_multiply_mode,
            fast_or_no_chroma_subsampling,
        )
    } else {
        yuv8_to_rgb_any(
            image,
            rgb,
            alpha_multiply_mode,
            fast_or_no_chroma_subsampling,
        )
    }
}

#[derive(Debug, Default)]
struct YUVBlock(f32, f32, f32);

pub(crate) fn rgb_gray_to_yuv(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<()> {
    let handled_y = {
        #[cfg(target_arch = "x86_64")]
        {
            avx2::try_rgb_gray_to_yuv(rgb, image)?.is_some()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };
    if !handled_y {
        let rgb_channel_count = rgb.channel_count() as usize;
        let gray_offset = rgb.format.r_offset();
        let rgb_max_channel_f = rgb.max_channel_f();
        let (bias_y, range_y) = bias_and_range_y(image);
        let yuv_max_channel = image.max_channel();
        let width = usize_from_u32(image.width)?;
        if image.depth == 8 {
            let mut dst_y = UninitPlane8::for_image(image, Plane::Y)?;
            for j in 0..image.height {
                let row_y = dst_y.spare_row_mut(j)?;
                if rgb.depth == 8 {
                    let src = rgb.row(j)?;
                    for (i, cell) in row_y.iter_mut().enumerate().take(width) {
                        let g =
                            src[(i * rgb_channel_count) + gray_offset] as f32 / rgb_max_channel_f;
                        // TODO: b/410088660 - handle alpha multiply/unmultiply.
                        cell.write(to_unorm(bias_y, range_y, yuv_max_channel, g) as u8);
                    }
                } else {
                    let src = rgb.row16(j)?;
                    for (i, cell) in row_y.iter_mut().enumerate().take(width) {
                        let g =
                            src[(i * rgb_channel_count) + gray_offset] as f32 / rgb_max_channel_f;
                        cell.write(to_unorm(bias_y, range_y, yuv_max_channel, g) as u8);
                    }
                }
            }
            // SAFETY: Every `(0..height) x (0..width)` Y sample was initialized above.
            unsafe { dst_y.finish_into(image, Plane::Y, 0)? };
        } else {
            let mut dst_y = UninitPlane16::for_image(image, Plane::Y)?;
            for j in 0..image.height {
                let row_y = dst_y.spare_row_mut(j)?;
                if rgb.depth == 8 {
                    let src = rgb.row(j)?;
                    for (i, cell) in row_y.iter_mut().enumerate().take(width) {
                        let g =
                            src[(i * rgb_channel_count) + gray_offset] as f32 / rgb_max_channel_f;
                        cell.write(to_unorm(bias_y, range_y, yuv_max_channel, g));
                    }
                } else {
                    let src = rgb.row16(j)?;
                    for (i, cell) in row_y.iter_mut().enumerate().take(width) {
                        let g =
                            src[(i * rgb_channel_count) + gray_offset] as f32 / rgb_max_channel_f;
                        cell.write(to_unorm(bias_y, range_y, yuv_max_channel, g));
                    }
                }
            }
            // SAFETY: Every `(0..height) x (0..width)` Y sample was initialized above.
            unsafe { dst_y.finish_into(image, Plane::Y, 0)? };
        }
    }
    let chroma_value = (image.max_channel() / 2) + 1;
    image.allocate_plane_with_default_value(Plane::U, chroma_value)?;
    image.allocate_plane_with_default_value(Plane::V, chroma_value)?;
    Ok(())
}

#[inline(always)]
fn rgb_pixel_to_yuv_pixel(
    mode: Mode,
    r: f32,
    g: f32,
    b: f32,
    rgb_max_channel_f: f32,
    range_y: f32,
    range_uv: f32,
) -> YUVBlock {
    match mode {
        Mode::YuvCoefficients(kr, kg, kb) => {
            let y = (kr * r) + (kg * g) + (kb * b);
            YUVBlock(
                y,
                (b - y) / (2.0 * (1.0 - kb)),
                (r - y) / (2.0 * (1.0 - kr)),
            )
        }
        Mode::Identity => {
            // Formulas 41,42,43 from https://www.itu.int/rec/T-REC-H.273-201612-S.
            YUVBlock(g, b, r)
        }
        Mode::YcgcoRe | Mode::YcgcoRo => {
            // Formulas 58,59,60,61 from https://www.itu.int/rec/T-REC-H.273-202407-P.
            let r = ((r * rgb_max_channel_f).clamp(0.0, rgb_max_channel_f) + 0.5) as i32;
            let g = ((g * rgb_max_channel_f).clamp(0.0, rgb_max_channel_f) + 0.5) as i32;
            let b = ((b * rgb_max_channel_f).clamp(0.0, rgb_max_channel_f) + 0.5) as i32;
            let co = r - b;
            let t = b + (co >> 1);
            let cg = g - t;
            YUVBlock(
                (t + (cg >> 1)) as f32 / range_y,
                cg as f32 / range_uv,
                co as f32 / range_uv,
            )
        }
        Mode::Ycgco => {
            // Formulas 44,45,46 from https://www.itu.int/rec/T-REC-H.273-201612-S.
            YUVBlock(
                0.5 * g + 0.25 * (r + b),
                0.5 * g - 0.25 * (r + b),
                0.5 * (r - b),
            )
        }
    }
}

// Reads an RGB pixel and converts it into YUV.
macro_rules! yuv_pixel {
    ($rgb:expr, $src16:expr, $src:expr, $i:expr, $rgb_channel_count:expr, $r_offset:expr, $g_offset:expr, $b_offset:expr, $rgb_max_channel_f:expr, $mode:expr, $range_y:expr, $range_uv:expr) => {{
        let base = $i * $rgb_channel_count;
        let rgb_pixel = if $rgb.depth > 8 {
            [
                $src16[base + $r_offset] as f32 / $rgb_max_channel_f,
                $src16[base + $g_offset] as f32 / $rgb_max_channel_f,
                $src16[base + $b_offset] as f32 / $rgb_max_channel_f,
            ]
        } else {
            [
                $src[base + $r_offset] as f32 / 255.0,
                $src[base + $g_offset] as f32 / 255.0,
                $src[base + $b_offset] as f32 / 255.0,
            ]
        };
        // TODO: b/410088660 - handle alpha multiply/unmultiply.
        rgb_pixel_to_yuv_pixel(
            $mode,
            rgb_pixel[0],
            rgb_pixel[1],
            rgb_pixel[2],
            $rgb_max_channel_f,
            $range_y,
            $range_uv,
        )
    }};
}

#[inline(always)]
fn plane_row_mut<'a>(
    plane: &'a mut Option<crate::utils::pixels::Pixels>,
    pd: &PlaneData,
    row: u32,
) -> AvifResult<&'a mut [u8]> {
    let start = checked_mul!(row, pd.row_bytes)?;
    plane
        .as_mut()
        .ok_or(AvifError::NoContent)?
        .slice_mut(start, pd.width)
}

#[inline(always)]
fn plane_row16_mut<'a>(
    plane: &'a mut Option<crate::utils::pixels::Pixels>,
    pd: &PlaneData,
    row: u32,
) -> AvifResult<&'a mut [u16]> {
    let start = checked_mul!(row, pd.row_bytes / 2)?;
    plane
        .as_mut()
        .ok_or(AvifError::NoContent)?
        .slice16_mut(start, pd.width)
}

#[inline(always)]
fn plane_two_rows_mut<'a>(
    plane: &'a mut Option<crate::utils::pixels::Pixels>,
    pd: &PlaneData,
    row: u32,
) -> AvifResult<(&'a mut [u8], &'a mut [u8])> {
    let start = checked_mul!(row, pd.row_bytes)?;
    let total_size = checked_add!(pd.row_bytes, pd.width)?;
    let both = plane
        .as_mut()
        .ok_or(AvifError::NoContent)?
        .slice_mut(start, total_size)?;
    let row_bytes = usize_from_u32(pd.row_bytes)?;
    let width = usize_from_u32(pd.width)?;
    let (first, row1) = both.split_at_mut(row_bytes);
    let row0 = first.get_mut(..width).ok_or(AvifError::NoContent)?;
    Ok((row0, row1))
}

#[inline(always)]
fn plane_two_rows16_mut<'a>(
    plane: &'a mut Option<crate::utils::pixels::Pixels>,
    pd: &PlaneData,
    row: u32,
) -> AvifResult<(&'a mut [u16], &'a mut [u16])> {
    let row_stride = pd.row_bytes / 2;
    let start = checked_mul!(row, row_stride)?;
    let total_size = checked_add!(row_stride, pd.width)?;
    let both = plane
        .as_mut()
        .ok_or(AvifError::NoContent)?
        .slice16_mut(start, total_size)?;
    let row_stride = usize_from_u32(row_stride)?;
    let width = usize_from_u32(pd.width)?;
    let (first, row1) = both.split_at_mut(row_stride);
    let row0 = first.get_mut(..width).ok_or(AvifError::NoContent)?;
    Ok((row0, row1))
}

fn rgb_to_yuv_420(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<()> {
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_max_channel_f = rgb.max_channel_f();
    let mode = (image as &image::Image).into();
    let (bias_y, range_y) = bias_and_range_y(image);
    let (bias_uv, range_uv) = if mode == Mode::Identity {
        (bias_y, range_y)
    } else {
        bias_and_range_uv(image)
    };
    let yuv_max_channel = image.max_channel();
    let width = image.width as usize;
    let height = image.height;
    let image_depth = image.depth;
    let y_pd = image.plane_data(Plane::Y).ok_or(AvifError::NoContent)?;
    let u_pd = image.plane_data(Plane::U).ok_or(AvifError::NoContent)?;
    let v_pd = image.plane_data(Plane::V).ok_or(AvifError::NoContent)?;
    let [plane_y, plane_u, plane_v, _] = &mut image.planes;

    for j in (0..height - 1).step_by(2) {
        let uv_j = j >> 1;
        let (dst_y16, dst_y16_1, dst_u16, dst_v16, dst_y, dst_y_1, dst_u, dst_v) =
            if image_depth > 8 {
                let (y0, y1) = plane_two_rows16_mut(plane_y, &y_pd, j)?;
                (
                    y0,
                    y1,
                    plane_row16_mut(plane_u, &u_pd, uv_j)?,
                    plane_row16_mut(plane_v, &v_pd, uv_j)?,
                    &mut [][..],
                    &mut [][..],
                    &mut [][..],
                    &mut [][..],
                )
            } else {
                let (y0, y1) = plane_two_rows_mut(plane_y, &y_pd, j)?;
                (
                    &mut [][..],
                    &mut [][..],
                    &mut [][..],
                    &mut [][..],
                    y0,
                    y1,
                    plane_row_mut(plane_u, &u_pd, uv_j)?,
                    plane_row_mut(plane_v, &v_pd, uv_j)?,
                )
            };
        let (src16, src16_1, src, src_1): (&[u16], &[u16], &[u8], &[u8]) = if rgb.depth > 8 {
            (rgb.row16(j)?, rgb.row16(j + 1)?, &[], &[])
        } else {
            (&[], &[], rgb.row(j)?, rgb.row(j + 1)?)
        };
        for i in (0..width - 1).step_by(2) {
            let yuv_pixel = [
                yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
                yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i + 1,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
                yuv_pixel!(
                    rgb,
                    src16_1,
                    src_1,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
                yuv_pixel!(
                    rgb,
                    src16_1,
                    src_1,
                    i + 1,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
            ];
            let avg_u = (yuv_pixel[0].1 + yuv_pixel[1].1 + yuv_pixel[2].1 + yuv_pixel[3].1) / 4.0;
            let avg_v = (yuv_pixel[0].2 + yuv_pixel[1].2 + yuv_pixel[2].2 + yuv_pixel[3].2) / 4.0;
            let uv_i = i >> 1;
            if image_depth > 8 {
                dst_y16[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[0].0);
                dst_y16[i + 1] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[1].0);
                dst_y16_1[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[2].0);
                dst_y16_1[i + 1] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[3].0);
                dst_u16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_u);
                dst_v16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_v);
            } else {
                dst_y[i] = to_unorm8!(bias_y, range_y, yuv_pixel[0].0);
                dst_y[i + 1] = to_unorm8!(bias_y, range_y, yuv_pixel[1].0);
                dst_y_1[i] = to_unorm8!(bias_y, range_y, yuv_pixel[2].0);
                dst_y_1[i + 1] = to_unorm8!(bias_y, range_y, yuv_pixel[3].0);
                dst_u[uv_i] = to_unorm8!(bias_uv, range_uv, avg_u);
                dst_v[uv_i] = to_unorm8!(bias_uv, range_uv, avg_v);
            }
        }
        // Last column.
        if width % 2 != 0 {
            let i = width - 1;
            let yuv_pixel = [
                yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
                yuv_pixel!(
                    rgb,
                    src16_1,
                    src_1,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
            ];
            let avg_u = (yuv_pixel[0].1 + yuv_pixel[1].1) / 2.0;
            let avg_v = (yuv_pixel[0].2 + yuv_pixel[1].2) / 2.0;
            let uv_i = i >> 1;
            if image_depth > 8 {
                dst_y16[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[0].0);
                dst_y16_1[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[1].0);
                dst_u16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_u);
                dst_v16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_v);
            } else {
                dst_y[i] = to_unorm8!(bias_y, range_y, yuv_pixel[0].0);
                dst_y_1[i] = to_unorm8!(bias_y, range_y, yuv_pixel[1].0);
                dst_u[uv_i] = to_unorm8!(bias_uv, range_uv, avg_u);
                dst_v[uv_i] = to_unorm8!(bias_uv, range_uv, avg_v);
            }
        }
    }
    // Last row.
    if height % 2 != 0 {
        let j = height - 1;
        let uv_j = j >> 1;
        let (dst_y16, dst_u16, dst_v16, dst_y, dst_u, dst_v) = if image_depth > 8 {
            (
                plane_row16_mut(plane_y, &y_pd, j)?,
                plane_row16_mut(plane_u, &u_pd, uv_j)?,
                plane_row16_mut(plane_v, &v_pd, uv_j)?,
                &mut [][..],
                &mut [][..],
                &mut [][..],
            )
        } else {
            (
                &mut [][..],
                &mut [][..],
                &mut [][..],
                plane_row_mut(plane_y, &y_pd, j)?,
                plane_row_mut(plane_u, &u_pd, uv_j)?,
                plane_row_mut(plane_v, &v_pd, uv_j)?,
            )
        };
        let (src16, src): (&[u16], &[u8]) =
            if rgb.depth > 8 { (rgb.row16(j)?, &[]) } else { (&[], rgb.row(j)?) };
        for i in (0..width).step_by(2) {
            let is_last_column = width == 1 || i > width - 2;
            let yuv_pixel = [
                yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
                if is_last_column {
                    YUVBlock(0.0, 0.0, 0.0)
                } else {
                    yuv_pixel!(
                        rgb,
                        src16,
                        src,
                        i + 1,
                        rgb_channel_count,
                        r_offset,
                        g_offset,
                        b_offset,
                        rgb_max_channel_f,
                        mode,
                        range_y,
                        range_uv
                    )
                },
            ];
            let (avg_u, avg_v) = if is_last_column {
                (yuv_pixel[0].1, yuv_pixel[0].2)
            } else {
                (
                    (yuv_pixel[0].1 + yuv_pixel[1].1) / 2.0,
                    (yuv_pixel[0].2 + yuv_pixel[1].2) / 2.0,
                )
            };
            let uv_i = i >> 1;
            if image_depth > 8 {
                dst_y16[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[0].0);
                if !is_last_column {
                    dst_y16[i + 1] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[1].0);
                }
                dst_u16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_u);
                dst_v16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_v);
            } else {
                dst_y[i] = to_unorm8!(bias_y, range_y, yuv_pixel[0].0);
                if !is_last_column {
                    dst_y[i + 1] = to_unorm8!(bias_y, range_y, yuv_pixel[1].0);
                }
                dst_u[uv_i] = to_unorm8!(bias_uv, range_uv, avg_u);
                dst_v[uv_i] = to_unorm8!(bias_uv, range_uv, avg_v);
            }
        }
    }
    Ok(())
}

fn rgb_to_yuv_422(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<()> {
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_max_channel_f = rgb.max_channel_f();
    let mode = (image as &image::Image).into();
    let (bias_y, range_y) = bias_and_range_y(image);
    let (bias_uv, range_uv) = if mode == Mode::Identity {
        (bias_y, range_y)
    } else {
        bias_and_range_uv(image)
    };
    let yuv_max_channel = image.max_channel();
    let width = image.width as usize;
    let height = image.height;
    let image_depth = image.depth;
    let y_pd = image.plane_data(Plane::Y).ok_or(AvifError::NoContent)?;
    let u_pd = image.plane_data(Plane::U).ok_or(AvifError::NoContent)?;
    let v_pd = image.plane_data(Plane::V).ok_or(AvifError::NoContent)?;
    let [plane_y, plane_u, plane_v, _] = &mut image.planes;

    for j in 0..height {
        let (src16, src): (&[u16], &[u8]) =
            if rgb.depth > 8 { (rgb.row16(j)?, &[]) } else { (&[], rgb.row(j)?) };
        let (dst_y16, dst_u16, dst_v16, dst_y, dst_u, dst_v) = if image_depth > 8 {
            (
                plane_row16_mut(plane_y, &y_pd, j)?,
                plane_row16_mut(plane_u, &u_pd, j)?,
                plane_row16_mut(plane_v, &v_pd, j)?,
                &mut [][..],
                &mut [][..],
                &mut [][..],
            )
        } else {
            (
                &mut [][..],
                &mut [][..],
                &mut [][..],
                plane_row_mut(plane_y, &y_pd, j)?,
                plane_row_mut(plane_u, &u_pd, j)?,
                plane_row_mut(plane_v, &v_pd, j)?,
            )
        };
        for i in (0..width - 1).step_by(2) {
            let yuv_pixel = [
                yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
                yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i + 1,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                ),
            ];
            let avg_u = (yuv_pixel[0].1 + yuv_pixel[1].1) / 2.0;
            let avg_v = (yuv_pixel[0].2 + yuv_pixel[1].2) / 2.0;
            let uv_i = i >> 1;
            if image_depth > 8 {
                dst_y16[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[0].0);
                dst_y16[i + 1] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel[1].0);
                dst_u16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_u);
                dst_v16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, avg_v);
            } else {
                dst_y[i] = to_unorm8!(bias_y, range_y, yuv_pixel[0].0);
                dst_y[i + 1] = to_unorm8!(bias_y, range_y, yuv_pixel[1].0);
                dst_u[uv_i] = to_unorm8!(bias_uv, range_uv, avg_u);
                dst_v[uv_i] = to_unorm8!(bias_uv, range_uv, avg_v);
            }
        }
        // Last column.
        if width % 2 != 0 {
            let i = width - 1;
            let yuv_pixel = yuv_pixel!(
                rgb,
                src16,
                src,
                i,
                rgb_channel_count,
                r_offset,
                g_offset,
                b_offset,
                rgb_max_channel_f,
                mode,
                range_y,
                range_uv
            );
            let uv_i = i >> 1;
            if image_depth > 8 {
                dst_y16[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel.0);
                dst_u16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, yuv_pixel.1);
                dst_v16[uv_i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, yuv_pixel.2);
            } else {
                dst_y[i] = to_unorm8!(bias_y, range_y, yuv_pixel.0);
                dst_u[uv_i] = to_unorm8!(bias_uv, range_uv, yuv_pixel.1);
                dst_v[uv_i] = to_unorm8!(bias_uv, range_uv, yuv_pixel.2);
            }
        }
    }
    Ok(())
}

fn can_fuse_alpha_base(rgb: &rgb::Image, image: &image::Image) -> bool {
    rgb.has_alpha()
        && !rgb.format.is_gray()
        && rgb.format != Format::Rgba1010102
        && matches!(rgb.depth, 8 | 10 | 12 | 16)
        && matches!(image.depth, 8 | 10 | 12 | 16)
        && rgb.width == image.width
        && rgb.height == image.height
        && rgb.has_valid_pixels()
}

fn can_fuse_alpha(rgb: &rgb::Image, image: &image::Image) -> bool {
    image.has_plane(Plane::A) && can_fuse_alpha_base(rgb, image)
}

fn rgb_to_yuv_444(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<bool> {
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let a_offset = rgb.format.alpha_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_max_channel_f = rgb.max_channel_f();
    let mode = (image as &image::Image).into();
    let (bias_y, range_y) = bias_and_range_y(image);
    let (bias_uv, range_uv) = if mode == Mode::Identity {
        (bias_y, range_y)
    } else {
        bias_and_range_uv(image)
    };
    let yuv_max_channel = image.max_channel();
    let width = image.width as usize;
    let height = image.height;
    let image_depth = image.depth;
    let same_alpha_depth = rgb.depth == image_depth;
    let y_pd = image.plane_data(Plane::Y).ok_or(AvifError::NoContent)?;
    let u_pd = image.plane_data(Plane::U).ok_or(AvifError::NoContent)?;
    let v_pd = image.plane_data(Plane::V).ok_or(AvifError::NoContent)?;
    let a_pd = if can_fuse_alpha(rgb, image) { image.plane_data(Plane::A) } else { None };
    let alpha_imported = a_pd.is_some();
    let [plane_y, plane_u, plane_v, plane_a] = &mut image.planes;

    for j in 0..height {
        let (src16, src): (&[u16], &[u8]) =
            if rgb.depth > 8 { (rgb.row16(j)?, &[]) } else { (&[], rgb.row(j)?) };
        if image_depth > 8 {
            let dst_y16 = plane_row16_mut(plane_y, &y_pd, j)?;
            let dst_u16 = plane_row16_mut(plane_u, &u_pd, j)?;
            let dst_v16 = plane_row16_mut(plane_v, &v_pd, j)?;
            let mut dst_a16 = match &a_pd {
                Some(pd) => Some(plane_row16_mut(plane_a, pd, j)?),
                None => None,
            };
            for i in 0..width {
                let yuv_pixel = yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                );
                dst_y16[i] = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel.0);
                dst_u16[i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, yuv_pixel.1);
                dst_v16[i] = to_unorm16!(bias_uv, range_uv, yuv_max_channel, yuv_pixel.2);
                if let Some(a16) = dst_a16.as_deref_mut() {
                    let raw_a = if rgb.depth > 8 {
                        src16[(i * rgb_channel_count) + a_offset]
                    } else {
                        src[(i * rgb_channel_count) + a_offset] as u16
                    };
                    a16[i] = if same_alpha_depth {
                        raw_a
                    } else {
                        rgb::Image::rescale_alpha_value(raw_a, rgb_max_channel_f, yuv_max_channel)
                    };
                }
            }
        } else {
            let dst_y = plane_row_mut(plane_y, &y_pd, j)?;
            let dst_u = plane_row_mut(plane_u, &u_pd, j)?;
            let dst_v = plane_row_mut(plane_v, &v_pd, j)?;
            let mut dst_a = match &a_pd {
                Some(pd) => Some(plane_row_mut(plane_a, pd, j)?),
                None => None,
            };
            for i in 0..width {
                let yuv_pixel = yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    range_uv
                );
                dst_y[i] = to_unorm8!(bias_y, range_y, yuv_pixel.0);
                dst_u[i] = to_unorm8!(bias_uv, range_uv, yuv_pixel.1);
                dst_v[i] = to_unorm8!(bias_uv, range_uv, yuv_pixel.2);
                if let Some(a8) = dst_a.as_deref_mut() {
                    let raw_a = if rgb.depth > 8 {
                        src16[(i * rgb_channel_count) + a_offset]
                    } else {
                        src[(i * rgb_channel_count) + a_offset] as u16
                    };
                    a8[i] = if same_alpha_depth {
                        raw_a as u8
                    } else {
                        rgb::Image::rescale_alpha_value(raw_a, rgb_max_channel_f, yuv_max_channel)
                            as u8
                    };
                }
            }
        }
    }
    Ok(alpha_imported)
}

fn rgb_to_yuv_400(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<()> {
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb_channel_count = rgb.channel_count() as usize;
    let rgb_max_channel_f = rgb.max_channel_f();
    let mode = (image as &image::Image).into();
    let (bias_y, range_y) = bias_and_range_y(image);
    let yuv_max_channel = image.max_channel();
    let width = image.width as usize;
    let height = image.height;
    let image_depth = image.depth;

    for j in 0..height {
        let (src16, src): (&[u16], &[u8]) =
            if rgb.depth > 8 { (rgb.row16(j)?, &[]) } else { (&[], rgb.row(j)?) };
        if image_depth > 8 {
            let dst_y16 = image.row16_mut(Plane::Y, j)?;
            for (i, dst_pixel) in dst_y16.iter_mut().enumerate().take(width) {
                let yuv_pixel = yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    0.0
                );
                *dst_pixel = to_unorm16!(bias_y, range_y, yuv_max_channel, yuv_pixel.0);
            }
        } else {
            let dst_y = image.row_mut(Plane::Y, j)?;
            for (i, dst_pixel) in dst_y.iter_mut().enumerate().take(width) {
                let yuv_pixel = yuv_pixel!(
                    rgb,
                    src16,
                    src,
                    i,
                    rgb_channel_count,
                    r_offset,
                    g_offset,
                    b_offset,
                    rgb_max_channel_f,
                    mode,
                    range_y,
                    0.0
                );
                *dst_pixel = to_unorm8!(bias_y, range_y, yuv_pixel.0);
            }
        }
    }
    Ok(())
}

fn rgb_to_yuv_scalar(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<bool> {
    match image.yuv_format {
        PixelFormat::Yuv420 => rgb_to_yuv_420(rgb, image).map(|()| false),
        PixelFormat::Yuv422 => rgb_to_yuv_422(rgb, image).map(|()| false),
        PixelFormat::Yuv444 => rgb_to_yuv_444(rgb, image),
        PixelFormat::Yuv400 => rgb_to_yuv_400(rgb, image).map(|()| false),
        _ => Err(AvifError::NotImplemented),
    }
}

fn rgb_to_yuv_scalar_fallback(
    rgb: &rgb::Image,
    image: &mut image::Image,
    alloc_alpha: bool,
) -> AvifResult<bool> {
    image.allocate_planes(Category::Color)?;
    if alloc_alpha && image.yuv_format == PixelFormat::Yuv444 && can_fuse_alpha_base(rgb, image) {
        image.allocate_planes(Category::Alpha)?;
    }
    rgb_to_yuv_scalar(rgb, image)
}

// Converts RGB(A) samples to YUV(A). Returns Ok(true) if Plane::A was also populated in-pass.
#[cfg(test)]
pub(crate) fn rgb_to_yuv(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<bool> {
    let alloc_alpha = image.has_plane(Plane::A);
    rgb_to_yuv_uninit_with_threads(rgb, image, alloc_alpha, 1)
}

pub(crate) fn rgb_to_yuv_uninit_with_threads(
    rgb: &rgb::Image,
    image: &mut image::Image,
    alloc_alpha: bool,
    threads: u32,
) -> AvifResult<bool> {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = threads;
    #[cfg(target_arch = "x86_64")]
    if let Some(alpha_imported) =
        avx2::try_rgb_to_yuv_uninit_with_threads(rgb, image, alloc_alpha, threads)?
    {
        return Ok(alpha_imported);
    }
    rgb_to_yuv_scalar_fallback(rgb, image, alloc_alpha)
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum RgbFmt {
        Rgba,
        Bgra,
        Argb,
        Abgr,
        Rgb,
        Bgr,
    }

    impl RgbFmt {
        fn from_format(format: Format) -> Option<Self> {
            match format {
                Format::Rgba => Some(Self::Rgba),
                Format::Bgra => Some(Self::Bgra),
                Format::Argb => Some(Self::Argb),
                Format::Abgr => Some(Self::Abgr),
                Format::Rgb => Some(Self::Rgb),
                Format::Bgr => Some(Self::Bgr),
                _ => None,
            }
        }

        #[inline(always)]
        fn channels(self) -> usize {
            match self {
                Self::Rgba | Self::Bgra | Self::Argb | Self::Abgr => 4,
                Self::Rgb | Self::Bgr => 3,
            }
        }

        #[inline(always)]
        fn offsets(self) -> (usize, usize, usize, usize) {
            match self {
                Self::Rgba => (0, 1, 2, 3),
                Self::Bgra => (2, 1, 0, 3),
                Self::Argb => (1, 2, 3, 0),
                Self::Abgr => (3, 2, 1, 0),
                Self::Rgb => (0, 1, 2, 0),
                Self::Bgr => (2, 1, 0, 0),
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Avx2Config {
        mode: Mode,
        rgb_depth: u8,
        image_depth: u8,
        rescale_alpha: bool,
        rgb_max_f: f32,
        src_div_f: f32,
        bias_y: f32,
        range_y: f32,
        bias_uv: f32,
        range_uv: f32,
        yuv_max: u16,
        kr: f32,
        kg: f32,
        kb: f32,
        denom_u: f32,
        denom_v: f32,
    }

    struct Avx2Params {
        mode: Mode,
        rgb_depth: u8,
        image_depth: u8,
        rescale_alpha: bool,
        rgb_max_f: f32,
        src_div_f: f32,
        bias_y: f32,
        range_y: f32,
        bias_uv: f32,
        range_uv: f32,
        yuv_max: u16,
        kr: f32,
        kg: f32,
        kb: f32,
        denom_u: f32,
        denom_v: f32,
        v_kr: __m256,
        v_kg: __m256,
        v_kb: __m256,
        v_denom_u: __m256,
        v_denom_v: __m256,
        v_rgb_max: __m256,
        v_src_div: __m256,
        v_yuv_max_f: __m256,
        v_bias_y: __m256,
        v_range_y: __m256,
        v_bias_uv: __m256,
        v_range_uv: __m256,
        v_half: __m256,
        v_quarter: __m256,
        mask_ff: __m256i,
        mask_ffff: __m256i,
        v_max_u16: __m256i,
        shuf8: __m256i,
        shuf16: __m256i,
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    unsafe fn make_shuf8(fmt: RgbFmt) -> __m256i {
        match fmt {
            RgbFmt::Rgba => _mm256_setzero_si256(),
            RgbFmt::Bgra => _mm256_setr_epi8(
                2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, //
                2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
            ),
            RgbFmt::Argb => _mm256_setr_epi8(
                1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, //
                1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
            ),
            RgbFmt::Abgr => _mm256_setr_epi8(
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, //
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
            ),
            RgbFmt::Rgb => _mm256_setr_epi8(
                0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1, //
                4, 5, 6, -1, 7, 8, 9, -1, 10, 11, 12, -1, 13, 14, 15, -1,
            ),
            RgbFmt::Bgr => _mm256_setr_epi8(
                2, 1, 0, -1, 5, 4, 3, -1, 8, 7, 6, -1, 11, 10, 9, -1, //
                6, 5, 4, -1, 9, 8, 7, -1, 12, 11, 10, -1, 15, 14, 13, -1,
            ),
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    unsafe fn make_shuf16(fmt: RgbFmt) -> __m256i {
        match fmt {
            RgbFmt::Rgba => _mm256_setzero_si256(),
            RgbFmt::Bgra => _mm256_setr_epi8(
                4, 5, 2, 3, 0, 1, 6, 7, 12, 13, 10, 11, 8, 9, 14, 15, //
                4, 5, 2, 3, 0, 1, 6, 7, 12, 13, 10, 11, 8, 9, 14, 15,
            ),
            RgbFmt::Argb => _mm256_setr_epi8(
                2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, //
                2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9,
            ),
            RgbFmt::Abgr => _mm256_setr_epi8(
                6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, //
                6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9,
            ),
            RgbFmt::Rgb => _mm256_setr_epi8(
                0, 1, 2, 3, 4, 5, -1, -1, 6, 7, 8, 9, 10, 11, -1, -1, //
                4, 5, 6, 7, 8, 9, -1, -1, 10, 11, 12, 13, 14, 15, -1, -1,
            ),
            RgbFmt::Bgr => _mm256_setr_epi8(
                4, 5, 2, 3, 0, 1, -1, -1, 10, 11, 8, 9, 6, 7, -1, -1, //
                8, 9, 6, 7, 4, 5, -1, -1, 14, 15, 12, 13, 10, 11, -1, -1,
            ),
        }
    }

    fn build_config(rgb: &rgb::Image, image: &image::Image) -> Avx2Config {
        let mode: Mode = image.into();
        let (bias_y, range_y) = bias_and_range_y(image);
        let (bias_uv, range_uv) = if mode == Mode::Identity {
            (bias_y, range_y)
        } else {
            bias_and_range_uv(image)
        };
        let (kr, kg, kb, denom_u, denom_v) = match mode {
            Mode::YuvCoefficients(kr, kg, kb) => (kr, kg, kb, 2.0 * (1.0 - kb), 2.0 * (1.0 - kr)),
            _ => (0.0, 0.0, 0.0, 1.0, 1.0),
        };
        let rgb_max_f = rgb.max_channel_f();
        let src_div_f = if rgb.depth > 8 { rgb_max_f } else { 255.0 };
        let yuv_max = image.max_channel();
        Avx2Config {
            mode,
            rgb_depth: rgb.depth,
            image_depth: image.depth,
            rescale_alpha: rgb.depth != image.depth,
            rgb_max_f,
            src_div_f,
            bias_y,
            range_y,
            bias_uv,
            range_uv,
            yuv_max,
            kr,
            kg,
            kb,
            denom_u,
            denom_v,
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    unsafe fn build_params(cfg: &Avx2Config, fmt: RgbFmt) -> Avx2Params {
        // SAFETY: Target feature `avx2` is enabled on this function.
        let (shuf8, shuf16) = unsafe { (make_shuf8(fmt), make_shuf16(fmt)) };
        Avx2Params {
            mode: cfg.mode,
            rgb_depth: cfg.rgb_depth,
            image_depth: cfg.image_depth,
            rescale_alpha: cfg.rescale_alpha,
            rgb_max_f: cfg.rgb_max_f,
            src_div_f: cfg.src_div_f,
            bias_y: cfg.bias_y,
            range_y: cfg.range_y,
            bias_uv: cfg.bias_uv,
            range_uv: cfg.range_uv,
            yuv_max: cfg.yuv_max,
            kr: cfg.kr,
            kg: cfg.kg,
            kb: cfg.kb,
            denom_u: cfg.denom_u,
            denom_v: cfg.denom_v,
            v_kr: _mm256_set1_ps(cfg.kr),
            v_kg: _mm256_set1_ps(cfg.kg),
            v_kb: _mm256_set1_ps(cfg.kb),
            v_denom_u: _mm256_set1_ps(cfg.denom_u),
            v_denom_v: _mm256_set1_ps(cfg.denom_v),
            v_rgb_max: _mm256_set1_ps(cfg.rgb_max_f),
            v_src_div: _mm256_set1_ps(cfg.src_div_f),
            v_yuv_max_f: _mm256_set1_ps(cfg.yuv_max as f32),
            v_bias_y: _mm256_set1_ps(cfg.bias_y),
            v_range_y: _mm256_set1_ps(cfg.range_y),
            v_bias_uv: _mm256_set1_ps(cfg.bias_uv),
            v_range_uv: _mm256_set1_ps(cfg.range_uv),
            v_half: _mm256_set1_ps(0.5),
            v_quarter: _mm256_set1_ps(0.25),
            mask_ff: _mm256_set1_epi32(0xFF),
            mask_ffff: _mm256_set1_epi32(0xFFFF),
            v_max_u16: _mm256_set1_epi16(cfg.yuv_max as i16),
            shuf8,
            shuf16,
        }
    }

    /// Loads 8 8-bit pixels as canonical `[R, G, B, A]` 32-bit lanes.
    ///
    /// # Safety
    /// Caller must ensure `avx2` is available and `ptr` is valid for reading 32 bytes when
    /// `KIND != 2` or 24 bytes when `KIND == 2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load8_raw_u8<const KIND: u8>(ptr: *const u8, shuf8: __m256i) -> __m256i {
        if KIND == 0 {
            // SAFETY: Caller verified `ptr` points to at least 32 readable bytes.
            unsafe { _mm256_loadu_si256(ptr.cast()) }
        } else if KIND == 1 {
            // SAFETY: Caller verified `ptr` points to at least 32 readable bytes.
            let raw = unsafe { _mm256_loadu_si256(ptr.cast()) };
            _mm256_shuffle_epi8(raw, shuf8)
        } else {
            // SAFETY: Caller verified `ptr` points to at least 24 readable bytes; `lo` reads
            // `[0..16]` and `hi` reads `[8..24]`, never reading past byte 24.
            let (lo, hi) = unsafe {
                (
                    _mm_loadu_si128(ptr.cast()),
                    _mm_loadu_si128(ptr.add(8).cast()),
                )
            };
            _mm256_shuffle_epi8(_mm256_set_m128i(hi, lo), shuf8)
        }
    }

    /// Loads 8 8-bit pixels and converts R, G, B to normalized IEEE-754 `f32` via real `vdivps`.
    ///
    /// # Safety
    /// Same safety requirements as [`load8_raw_u8`].
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load8_f32_u8<const KIND: u8>(
        ptr: *const u8,
        p: &Avx2Params,
    ) -> (__m256, __m256, __m256, __m256i) {
        // SAFETY: Caller upholds `ptr` bounds and `avx2` availability.
        let px = unsafe { load8_raw_u8::<KIND>(ptr, p.shuf8) };
        let ri = _mm256_and_si256(px, p.mask_ff);
        let gi = _mm256_and_si256(_mm256_srli_epi32(px, 8), p.mask_ff);
        let bi = if KIND == 2 {
            _mm256_srli_epi32(px, 16)
        } else {
            _mm256_and_si256(_mm256_srli_epi32(px, 16), p.mask_ff)
        };
        let ai = _mm256_srli_epi32(px, 24);
        let r = _mm256_div_ps(_mm256_cvtepi32_ps(ri), p.v_src_div);
        let g = _mm256_div_ps(_mm256_cvtepi32_ps(gi), p.v_src_div);
        let b = _mm256_div_ps(_mm256_cvtepi32_ps(bi), p.v_src_div);
        (r, g, b, ai)
    }

    /// Loads 8 >8-bit (`u16`) pixels and converts R, G, B to normalized IEEE-754 `f32`.
    ///
    /// # Safety
    /// Caller must ensure `avx2` is available and `ptr` is valid for reading 32 `u16` elements
    /// (64 bytes) when `KIND != 2` or 24 `u16` elements (48 bytes) when `KIND == 2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load8_f32_u16<const KIND: u8>(
        ptr: *const u16,
        p: &Avx2Params,
    ) -> (__m256, __m256, __m256, __m256i) {
        let (p0, p1) = if KIND == 0 {
            // SAFETY: Caller verified `ptr` points to at least 32 readable `u16`s (64 bytes).
            unsafe {
                (
                    _mm256_loadu_si256(ptr.cast()),
                    _mm256_loadu_si256(ptr.add(16).cast()),
                )
            }
        } else if KIND == 1 {
            // SAFETY: Caller verified `ptr` points to at least 32 readable `u16`s (64 bytes).
            let (r0, r1) = unsafe {
                (
                    _mm256_loadu_si256(ptr.cast()),
                    _mm256_loadu_si256(ptr.add(16).cast()),
                )
            };
            (
                _mm256_shuffle_epi8(r0, p.shuf16),
                _mm256_shuffle_epi8(r1, p.shuf16),
            )
        } else {
            let bp = ptr as *const u8;
            // SAFETY: Caller verified `ptr` points to at least 24 readable `u16`s (48 bytes);
            // the four 16-byte loads read `[0..16]`, `[8..24]`, `[24..40]`, and `[32..48]`.
            let (l0, h0, l1, h1) = unsafe {
                (
                    _mm_loadu_si128(bp.cast()),
                    _mm_loadu_si128(bp.add(8).cast()),
                    _mm_loadu_si128(bp.add(24).cast()),
                    _mm_loadu_si128(bp.add(32).cast()),
                )
            };
            (
                _mm256_shuffle_epi8(_mm256_set_m128i(h0, l0), p.shuf16),
                _mm256_shuffle_epi8(_mm256_set_m128i(h1, l1), p.shuf16),
            )
        };
        let p0f = _mm256_castsi256_ps(p0);
        let p1f = _mm256_castsi256_ps(p1);
        let rg = _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(_mm256_shuffle_ps::<0x88>(
            p0f, p1f,
        )));
        let ba = _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(
            p0f, p1f,
        )));
        let ri = _mm256_and_si256(rg, p.mask_ffff);
        let gi = _mm256_srli_epi32(rg, 16);
        let bi = if KIND == 2 { ba } else { _mm256_and_si256(ba, p.mask_ffff) };
        let ai = _mm256_srli_epi32(ba, 16);
        let r = _mm256_div_ps(_mm256_cvtepi32_ps(ri), p.v_src_div);
        let g = _mm256_div_ps(_mm256_cvtepi32_ps(gi), p.v_src_div);
        let b = _mm256_div_ps(_mm256_cvtepi32_ps(bi), p.v_src_div);
        (r, g, b, ai)
    }

    const MODE_COEFFS: u8 = 0;
    const MODE_IDENTITY: u8 = 1;
    const MODE_YCGCO: u8 = 2;
    const MODE_YCGCO_RE_RO: u8 = 3;

    #[inline(always)]
    fn mode_tag(mode: Mode) -> u8 {
        match mode {
            Mode::YuvCoefficients(..) => MODE_COEFFS,
            Mode::Identity => MODE_IDENTITY,
            Mode::Ycgco => MODE_YCGCO,
            Mode::YcgcoRe | Mode::YcgcoRo => MODE_YCGCO_RE_RO,
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn yuv_coeffs_pair(
        p: &Avx2Params,
        r0: __m256,
        g0: __m256,
        b0: __m256,
        r1: __m256,
        g1: __m256,
        b1: __m256,
    ) -> ((__m256, __m256, __m256), (__m256, __m256, __m256)) {
        let y0 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p.v_kr, r0), _mm256_mul_ps(p.v_kg, g0)),
            _mm256_mul_ps(p.v_kb, b0),
        );
        let y1 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p.v_kr, r1), _mm256_mul_ps(p.v_kg, g1)),
            _mm256_mul_ps(p.v_kb, b1),
        );
        let u0 = _mm256_div_ps(_mm256_sub_ps(b0, y0), p.v_denom_u);
        let u1 = _mm256_div_ps(_mm256_sub_ps(b1, y1), p.v_denom_u);
        let v0 = _mm256_div_ps(_mm256_sub_ps(r0, y0), p.v_denom_v);
        let v1 = _mm256_div_ps(_mm256_sub_ps(r1, y1), p.v_denom_v);
        ((y0, u0, v0), (y1, u1, v1))
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn yuv_one(p: &Avx2Params, r: __m256, g: __m256, b: __m256) -> (__m256, __m256, __m256) {
        let y = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p.v_kr, r), _mm256_mul_ps(p.v_kg, g)),
            _mm256_mul_ps(p.v_kb, b),
        );
        let u = _mm256_div_ps(_mm256_sub_ps(b, y), p.v_denom_u);
        let v = _mm256_div_ps(_mm256_sub_ps(r, y), p.v_denom_v);
        (y, u, v)
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn yuv_one_mode<const MODE_TAG: u8>(
        p: &Avx2Params,
        r: __m256,
        g: __m256,
        b: __m256,
    ) -> (__m256, __m256, __m256) {
        if MODE_TAG == MODE_COEFFS {
            // SAFETY: `avx2` is enabled on this function.
            unsafe { yuv_one(p, r, g, b) }
        } else if MODE_TAG == MODE_IDENTITY {
            (g, b, r)
        } else if MODE_TAG == MODE_YCGCO {
            let g50 = _mm256_mul_ps(p.v_half, g);
            let rb25 = _mm256_mul_ps(p.v_quarter, _mm256_add_ps(r, b));
            (
                _mm256_add_ps(g50, rb25),
                _mm256_sub_ps(g50, rb25),
                _mm256_mul_ps(p.v_half, _mm256_sub_ps(r, b)),
            )
        } else {
            let zero = _mm256_setzero_ps();
            let ri = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_min_ps(
                    _mm256_max_ps(_mm256_mul_ps(r, p.v_rgb_max), zero),
                    p.v_rgb_max,
                ),
                p.v_half,
            ));
            let gi = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_min_ps(
                    _mm256_max_ps(_mm256_mul_ps(g, p.v_rgb_max), zero),
                    p.v_rgb_max,
                ),
                p.v_half,
            ));
            let bi = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_min_ps(
                    _mm256_max_ps(_mm256_mul_ps(b, p.v_rgb_max), zero),
                    p.v_rgb_max,
                ),
                p.v_half,
            ));
            let co = _mm256_sub_epi32(ri, bi);
            let t = _mm256_add_epi32(bi, _mm256_srai_epi32(co, 1));
            let cg = _mm256_sub_epi32(gi, t);
            let y = _mm256_div_ps(
                _mm256_cvtepi32_ps(_mm256_add_epi32(t, _mm256_srai_epi32(cg, 1))),
                p.v_range_y,
            );
            let u = _mm256_div_ps(_mm256_cvtepi32_ps(cg), p.v_range_uv);
            let v = _mm256_div_ps(_mm256_cvtepi32_ps(co), p.v_range_uv);
            (y, u, v)
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn yuv_pair_mode<const MODE_TAG: u8>(
        p: &Avx2Params,
        r0: __m256,
        g0: __m256,
        b0: __m256,
        r1: __m256,
        g1: __m256,
        b1: __m256,
    ) -> ((__m256, __m256, __m256), (__m256, __m256, __m256)) {
        if MODE_TAG == MODE_COEFFS {
            // SAFETY: `avx2` is enabled on this function.
            unsafe { yuv_coeffs_pair(p, r0, g0, b0, r1, g1, b1) }
        } else {
            // SAFETY: `avx2` is enabled on this function.
            unsafe {
                (
                    yuv_one_mode::<MODE_TAG>(p, r0, g0, b0),
                    yuv_one_mode::<MODE_TAG>(p, r1, g1, b1),
                )
            }
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn luma_one_mode<const MODE_TAG: u8>(
        p: &Avx2Params,
        r: __m256,
        g: __m256,
        b: __m256,
    ) -> __m256 {
        if MODE_TAG == MODE_COEFFS {
            _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(p.v_kr, r), _mm256_mul_ps(p.v_kg, g)),
                _mm256_mul_ps(p.v_kb, b),
            )
        } else if MODE_TAG == MODE_IDENTITY {
            g
        } else if MODE_TAG == MODE_YCGCO {
            _mm256_add_ps(
                _mm256_mul_ps(p.v_half, g),
                _mm256_mul_ps(p.v_quarter, _mm256_add_ps(r, b)),
            )
        } else {
            let zero = _mm256_setzero_ps();
            let ri = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_min_ps(
                    _mm256_max_ps(_mm256_mul_ps(r, p.v_rgb_max), zero),
                    p.v_rgb_max,
                ),
                p.v_half,
            ));
            let gi = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_min_ps(
                    _mm256_max_ps(_mm256_mul_ps(g, p.v_rgb_max), zero),
                    p.v_rgb_max,
                ),
                p.v_half,
            ));
            let bi = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_min_ps(
                    _mm256_max_ps(_mm256_mul_ps(b, p.v_rgb_max), zero),
                    p.v_rgb_max,
                ),
                p.v_half,
            ));
            let co = _mm256_sub_epi32(ri, bi);
            let t = _mm256_add_epi32(bi, _mm256_srai_epi32(co, 1));
            let cg = _mm256_sub_epi32(gi, t);
            _mm256_div_ps(
                _mm256_cvtepi32_ps(_mm256_add_epi32(t, _mm256_srai_epi32(cg, 1))),
                p.v_range_y,
            )
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quant_i32(val: __m256, v_range: __m256, v_bias: __m256, v_half: __m256) -> __m256i {
        let scaled = _mm256_add_ps(_mm256_mul_ps(val, v_range), v_bias);
        _mm256_cvttps_epi32(_mm256_add_ps(v_half, scaled))
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 16 bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store16_u8_ptr(dst: *mut u8, lo: __m256i, hi: __m256i) {
        let w16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo, hi));
        let b8 = _mm_packus_epi16(
            _mm256_castsi256_si128(w16),
            _mm256_extracti128_si256::<1>(w16),
        );
        // SAFETY: Caller guarantees `dst` points to 16 writable bytes.
        unsafe { _mm_storeu_si128(dst.cast(), b8) };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 8 bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store8_u8_ptr(dst: *mut u8, lo: __m256i) {
        let w16 = _mm_packs_epi32(
            _mm256_castsi256_si128(lo),
            _mm256_extracti128_si256::<1>(lo),
        );
        let b8 = _mm_packus_epi16(w16, w16);
        // SAFETY: Caller guarantees `dst` points to 8 writable bytes.
        unsafe { _mm_storel_epi64(dst.cast(), b8) };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 16 `u16`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store16_u16_ptr(dst: *mut u16, lo: __m256i, hi: __m256i, v_max_u16: __m256i) {
        let u16_ord = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi));
        let clamped = _mm256_min_epu16(u16_ord, v_max_u16);
        // SAFETY: Caller guarantees `dst` points to 32 writable bytes.
        unsafe { _mm256_storeu_si256(dst.cast(), clamped) };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 8 `u16`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store8_u16_ptr(dst: *mut u16, lo: __m256i, v_max_u16: __m256i) {
        let u16x8 = _mm_packus_epi32(
            _mm256_castsi256_si128(lo),
            _mm256_extracti128_si256::<1>(lo),
        );
        let clamped = _mm_min_epu16(u16x8, _mm256_castsi256_si128(v_max_u16));
        // SAFETY: Caller guarantees `dst` points to 16 writable bytes.
        unsafe { _mm_storeu_si128(dst.cast(), clamped) };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 16 `u16`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store16_alpha_u16_ptr(dst: *mut u16, lo: __m256i, hi: __m256i) {
        let u16_ord = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi));
        // SAFETY: Caller guarantees `dst` points to 32 writable bytes.
        unsafe { _mm256_storeu_si256(dst.cast(), u16_ord) };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 8 `u16`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store8_alpha_u16_ptr(dst: *mut u16, lo: __m256i) {
        let u16x8 = _mm_packus_epi32(
            _mm256_castsi256_si128(lo),
            _mm256_extracti128_si256::<1>(lo),
        );
        // SAFETY: Caller guarantees `dst` points to 16 writable bytes.
        unsafe { _mm_storeu_si128(dst.cast(), u16x8) };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rescale_alpha_i32(ai: __m256i, p: &Avx2Params) -> __m256i {
        let alpha_f = _mm256_div_ps(_mm256_cvtepi32_ps(ai), p.v_src_div);
        let scaled = _mm256_mul_ps(alpha_f, p.v_yuv_max_f);
        _mm256_cvttps_epi32(_mm256_add_ps(p.v_half, scaled))
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 16 `u16`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store16_fused_alpha_u16(dst: *mut u16, a0: __m256i, a1: __m256i, p: &Avx2Params) {
        if p.rescale_alpha {
            // SAFETY: Caller guarantees `avx2` and 16 writable `u16`s at `dst`.
            unsafe {
                let q0 = rescale_alpha_i32(a0, p);
                let q1 = rescale_alpha_i32(a1, p);
                store16_u16_ptr(dst, q0, q1, p.v_max_u16);
            }
        } else {
            // SAFETY: Caller guarantees `avx2` and 16 writable `u16`s at `dst`.
            unsafe { store16_alpha_u16_ptr(dst, a0, a1) };
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 8 `u16`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store8_fused_alpha_u16(dst: *mut u16, a0: __m256i, p: &Avx2Params) {
        if p.rescale_alpha {
            // SAFETY: Caller guarantees `avx2` and 8 writable `u16`s at `dst`.
            unsafe {
                let q0 = rescale_alpha_i32(a0, p);
                store8_u16_ptr(dst, q0, p.v_max_u16);
            }
        } else {
            // SAFETY: Caller guarantees `avx2` and 8 writable `u16`s at `dst`.
            unsafe { store8_alpha_u16_ptr(dst, a0) };
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 16 `u8`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store16_fused_alpha_u8(dst: *mut u8, a0: __m256i, a1: __m256i, p: &Avx2Params) {
        if p.rescale_alpha {
            // SAFETY: Caller guarantees `avx2` and 16 writable `u8`s at `dst`.
            unsafe {
                let q0 = rescale_alpha_i32(a0, p);
                let q1 = rescale_alpha_i32(a1, p);
                store16_u8_ptr(dst, q0, q1);
            }
        } else {
            // SAFETY: Caller guarantees `avx2` and 16 writable `u8`s at `dst`.
            unsafe { store16_u8_ptr(dst, a0, a1) };
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 8 `u8`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store8_fused_alpha_u8(dst: *mut u8, a0: __m256i, p: &Avx2Params) {
        if p.rescale_alpha {
            // SAFETY: Caller guarantees `avx2` and 8 writable `u8`s at `dst`.
            unsafe {
                let q0 = rescale_alpha_i32(a0, p);
                store8_u8_ptr(dst, q0);
            }
        } else {
            // SAFETY: Caller guarantees `avx2` and 8 writable `u8`s at `dst`.
            unsafe { store8_u8_ptr(dst, a0) };
        }
    }

    /// # Safety
    /// The caller must ensure `dst` is writable for 1 `u16`.
    #[inline]
    unsafe fn store1_fused_alpha_u16(dst: *mut u16, a: u16, p: &Avx2Params) {
        let val = if p.rescale_alpha {
            rgb::Image::rescale_alpha_value(a, p.rgb_max_f, p.yuv_max)
        } else {
            a
        };
        // SAFETY: Caller guarantees `dst` points to 1 writable `u16`.
        unsafe { *dst = val };
    }

    /// # Safety
    /// The caller must ensure `dst` is writable for 1 `u8`.
    #[inline]
    unsafe fn store1_fused_alpha_u8(dst: *mut u8, a: u16, p: &Avx2Params) {
        let val = if p.rescale_alpha {
            rgb::Image::rescale_alpha_value(a, p.rgb_max_f, p.yuv_max) as u8
        } else {
            a as u8
        };
        // SAFETY: Caller guarantees `dst` points to 1 writable `u8`.
        unsafe { *dst = val };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quant_avg16_422(c_lo: __m256, c_hi: __m256, p: &Avx2Params) -> __m256i {
        let sum = _mm256_add_ps(
            _mm256_shuffle_ps::<0x88>(c_lo, c_hi),
            _mm256_shuffle_ps::<0xDD>(c_lo, c_hi),
        );
        let avg = _mm256_mul_ps(sum, p.v_half);
        // SAFETY: `avx2` is enabled on this function.
        let qi = unsafe { quant_i32(avg, p.v_range_uv, p.v_bias_uv, p.v_half) };
        _mm256_permute4x64_epi64::<0xD8>(qi)
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quant_avg8_422(c: __m256, p: &Avx2Params) -> __m128i {
        let lo = _mm256_castps256_ps128(c);
        let hi = _mm256_extractf128_ps::<1>(c);
        let sum = _mm_add_ps(
            _mm_shuffle_ps::<0x88>(lo, hi),
            _mm_shuffle_ps::<0xDD>(lo, hi),
        );
        let avg = _mm_mul_ps(sum, _mm256_castps256_ps128(p.v_half));
        let scaled = _mm_add_ps(
            _mm_mul_ps(avg, _mm256_castps256_ps128(p.v_range_uv)),
            _mm256_castps256_ps128(p.v_bias_uv),
        );
        _mm_cvttps_epi32(_mm_add_ps(_mm256_castps256_ps128(p.v_half), scaled))
    }

    /// Preserves exact scalar left-to-right sum `(((c00 + c01) + c10) + c11) / 4.0`.
    ///
    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quant_avg16_420(
        c0_lo: __m256,
        c0_hi: __m256,
        c1_lo: __m256,
        c1_hi: __m256,
        p: &Avx2Params,
    ) -> __m256i {
        let e0 = _mm256_shuffle_ps::<0x88>(c0_lo, c0_hi);
        let o0 = _mm256_shuffle_ps::<0xDD>(c0_lo, c0_hi);
        let e1 = _mm256_shuffle_ps::<0x88>(c1_lo, c1_hi);
        let o1 = _mm256_shuffle_ps::<0xDD>(c1_lo, c1_hi);
        let s01 = _mm256_add_ps(e0, o0);
        let s012 = _mm256_add_ps(s01, e1);
        let s0123 = _mm256_add_ps(s012, o1);
        let avg = _mm256_mul_ps(s0123, p.v_quarter);
        // SAFETY: `avx2` is enabled on this function.
        let qi = unsafe { quant_i32(avg, p.v_range_uv, p.v_bias_uv, p.v_half) };
        _mm256_permute4x64_epi64::<0xD8>(qi)
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn quant_avg8_420(c0: __m256, c1: __m256, p: &Avx2Params) -> __m128i {
        let lo0 = _mm256_castps256_ps128(c0);
        let hi0 = _mm256_extractf128_ps::<1>(c0);
        let lo1 = _mm256_castps256_ps128(c1);
        let hi1 = _mm256_extractf128_ps::<1>(c1);
        let e0 = _mm_shuffle_ps::<0x88>(lo0, hi0);
        let o0 = _mm_shuffle_ps::<0xDD>(lo0, hi0);
        let e1 = _mm_shuffle_ps::<0x88>(lo1, hi1);
        let o1 = _mm_shuffle_ps::<0xDD>(lo1, hi1);
        let avg = _mm_mul_ps(
            _mm_add_ps(_mm_add_ps(_mm_add_ps(e0, o0), e1), o1),
            _mm256_castps256_ps128(p.v_quarter),
        );
        let scaled = _mm_add_ps(
            _mm_mul_ps(avg, _mm256_castps256_ps128(p.v_range_uv)),
            _mm256_castps256_ps128(p.v_bias_uv),
        );
        _mm_cvttps_epi32(_mm_add_ps(_mm256_castps256_ps128(p.v_half), scaled))
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 4 bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store4_u8_ptr(dst: *mut u8, qi4: __m128i) {
        let w16 = _mm_packs_epi32(qi4, qi4);
        let b8 = _mm_packus_epi16(w16, w16);
        let bytes = _mm_cvtsi128_si32(b8).to_ne_bytes();
        // SAFETY: Caller guarantees `dst` points to 4 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, 4) };
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2` and `dst` is writable for 4 `u16`s.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store4_u16_ptr(dst: *mut u16, qi4: __m128i, v_max_u16: __m256i) {
        let u16x4 = _mm_min_epu16(
            _mm_packus_epi32(qi4, qi4),
            _mm256_castsi256_si128(v_max_u16),
        );
        // SAFETY: Caller guarantees `dst` points to 8 writable bytes.
        unsafe { _mm_storel_epi64(dst.cast(), u16x4) };
    }

    #[inline(always)]
    fn rgb_pixel_to_yuv_pixel_spec<const MODE_TAG: u8>(
        r: f32,
        g: f32,
        b: f32,
        p: &Avx2Params,
    ) -> YUVBlock {
        if MODE_TAG == MODE_COEFFS {
            debug_assert!(matches!(p.mode, Mode::YuvCoefficients(..)));
            let y = (p.kr * r) + (p.kg * g) + (p.kb * b);
            YUVBlock(y, (b - y) / p.denom_u, (r - y) / p.denom_v)
        } else if MODE_TAG == MODE_IDENTITY {
            YUVBlock(g, b, r)
        } else if MODE_TAG == MODE_YCGCO {
            YUVBlock(
                0.5 * g + 0.25 * (r + b),
                0.5 * g - 0.25 * (r + b),
                0.5 * (r - b),
            )
        } else {
            let ri = ((r * p.rgb_max_f).clamp(0.0, p.rgb_max_f) + 0.5) as i32;
            let gi = ((g * p.rgb_max_f).clamp(0.0, p.rgb_max_f) + 0.5) as i32;
            let bi = ((b * p.rgb_max_f).clamp(0.0, p.rgb_max_f) + 0.5) as i32;
            let co = ri - bi;
            let t = bi + (co >> 1);
            let cg = gi - t;
            YUVBlock(
                (t + (cg >> 1)) as f32 / p.range_y,
                cg as f32 / p.range_uv,
                co as f32 / p.range_uv,
            )
        }
    }

    /// # Safety
    /// Caller must ensure `src8` (when `RGB_8`) or `src16` (when `!RGB_8`) has at least
    /// `(i + 1) * fmt.channels()` readable elements.
    #[inline]
    unsafe fn scalar_px_spec<const MODE_TAG: u8, const RGB_8: bool>(
        src8: *const u8,
        src16: *const u16,
        i: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> (YUVBlock, u16) {
        let ch = fmt.channels();
        let (ro, go, bo, ao) = fmt.offsets();
        let base = i * ch;
        let (r, g, b, a) = if !RGB_8 {
            // SAFETY: Caller verified `src16` has at least `(i + 1) * ch` readable `u16` elements.
            unsafe {
                (
                    *src16.add(base + ro) as f32 / p.src_div_f,
                    *src16.add(base + go) as f32 / p.src_div_f,
                    *src16.add(base + bo) as f32 / p.src_div_f,
                    *src16.add(base + ao),
                )
            }
        } else {
            // SAFETY: Caller verified `src8` has at least `(i + 1) * ch` readable `u8` elements.
            unsafe {
                (
                    *src8.add(base + ro) as f32 / 255.0,
                    *src8.add(base + go) as f32 / 255.0,
                    *src8.add(base + bo) as f32 / 255.0,
                    *src8.add(base + ao) as u16,
                )
            }
        };
        (rgb_pixel_to_yuv_pixel_spec::<MODE_TAG>(r, g, b, p), a)
    }

    /// Dedicated pure-integer 8-bit Full-Range `Mode::Identity` `Yuv444` row kernel.
    ///
    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    unsafe fn row_444_identity_u8<const KIND: u8, const FUSE_A: bool>(
        src: &[u8],
        dst_y: &mut [MaybeUninit<u8>],
        dst_u: &mut [MaybeUninit<u8>],
        dst_v: &mut [MaybeUninit<u8>],
        dst_a: Option<&mut [MaybeUninit<u8>]>,
        width: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        if ch != if KIND == 2 { 3 } else { 4 } {
            return AvifError::no_content();
        }
        let src_req = width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        if src.len() < src_req || dst_y.len() < width || dst_u.len() < width || dst_v.len() < width
        {
            return AvifError::no_content();
        }
        let ap = if KIND != 2 && FUSE_A {
            let Some(a) = dst_a else {
                return AvifError::no_content();
            };
            if a.len() < width {
                return AvifError::no_content();
            }
            a.as_mut_ptr().cast::<u8>()
        } else {
            std::ptr::null_mut()
        };
        let sp = src.as_ptr();
        let yp = dst_y.as_mut_ptr().cast::<u8>();
        let up = dst_u.as_mut_ptr().cast::<u8>();
        let vp = dst_v.as_mut_ptr().cast::<u8>();

        let deint = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, //
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        );
        let shuf_deint = if KIND == 0 { deint } else { _mm256_shuffle_epi8(p.shuf8, deint) };
        let mut i = 0usize;
        while i + 16 <= width {
            // SAFETY: Entry bounds checks guarantee `src` has `(i + 16) * ch` readable bytes.
            let (raw0, raw1) = unsafe {
                let s_off = sp.add(i * ch);
                if KIND != 2 {
                    (
                        _mm256_loadu_si256(s_off.cast()),
                        _mm256_loadu_si256(s_off.add(32).cast()),
                    )
                } else {
                    let l0 = _mm_loadu_si128(s_off.cast());
                    let h0 = _mm_loadu_si128(s_off.add(8).cast());
                    let l1 = _mm_loadu_si128(s_off.add(24).cast());
                    let h1 = _mm_loadu_si128(s_off.add(32).cast());
                    (_mm256_set_m128i(h0, l0), _mm256_set_m128i(h1, l1))
                }
            };
            let s0 = _mm256_shuffle_epi8(raw0, shuf_deint);
            let s1 = _mm256_shuffle_epi8(raw1, shuf_deint);
            let lo = _mm256_permute2x128_si256::<0x20>(s0, s1);
            let hi = _mm256_permute2x128_si256::<0x31>(s0, s1);
            let rg = _mm256_unpacklo_epi32(lo, hi);
            let ba = _mm256_unpackhi_epi32(lo, hi);
            let rg_lo = _mm256_castsi256_si128(rg);
            let rg_hi = _mm256_extracti128_si256::<1>(rg);
            let ba_lo = _mm256_castsi256_si128(ba);
            let ba_hi = _mm256_extracti128_si256::<1>(ba);
            let v_out = _mm_unpacklo_epi64(rg_lo, rg_hi);
            let y_out = _mm_unpackhi_epi64(rg_lo, rg_hi);
            let u_out = _mm_unpacklo_epi64(ba_lo, ba_hi);
            // SAFETY: Entry bounds checks guarantee `yp`, `up`, `vp` have `width >= i + 16` bytes.
            unsafe {
                _mm_storeu_si128(yp.add(i).cast(), y_out);
                _mm_storeu_si128(up.add(i).cast(), u_out);
                _mm_storeu_si128(vp.add(i).cast(), v_out);
            }
            if KIND != 2 && FUSE_A {
                let a_out = _mm_unpackhi_epi64(ba_lo, ba_hi);
                // SAFETY: Entry bounds check verified `ap` has `width >= i + 16` writable bytes.
                unsafe { _mm_storeu_si128(ap.add(i).cast(), a_out) };
            }
            i += 16;
        }
        let (ro, go, bo, ao) = fmt.offsets();
        for x in i..width {
            let base = x * ch;
            // SAFETY: `x < width` and entry checks guarantee `sp`, `yp`, `up`, `vp`, `ap` bounds.
            unsafe {
                *yp.add(x) = *sp.add(base + go);
                *up.add(x) = *sp.add(base + bo);
                *vp.add(x) = *sp.add(base + ro);
                if KIND != 2 && FUSE_A {
                    *ap.add(x) = *sp.add(base + ao);
                }
            }
        }
        Ok(())
    }

    /// Hot 8-bit -> 8-bit `Mode::YuvCoefficients` `Yuv444` row kernel with fused alpha extraction.
    ///
    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    unsafe fn row_444_u8_coeffs<const KIND: u8, const FUSE_A: bool>(
        src: &[u8],
        dst_y: &mut [MaybeUninit<u8>],
        dst_u: &mut [MaybeUninit<u8>],
        dst_v: &mut [MaybeUninit<u8>],
        dst_a: Option<&mut [MaybeUninit<u8>]>,
        width: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        if ch != if KIND == 2 { 3 } else { 4 } {
            return AvifError::no_content();
        }
        let src_req = width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        if src.len() < src_req || dst_y.len() < width || dst_u.len() < width || dst_v.len() < width
        {
            return AvifError::no_content();
        }
        let ap = if KIND != 2 && FUSE_A {
            let Some(a) = dst_a else {
                return AvifError::no_content();
            };
            if a.len() < width {
                return AvifError::no_content();
            }
            a.as_mut_ptr().cast::<u8>()
        } else {
            std::ptr::null_mut()
        };
        let sp = src.as_ptr();
        let yp = dst_y.as_mut_ptr().cast::<u8>();
        let up = dst_u.as_mut_ptr().cast::<u8>();
        let vp = dst_v.as_mut_ptr().cast::<u8>();

        let mut i = 0usize;
        while i + 16 <= width {
            // SAFETY: Entry bounds checks guarantee `src` has `(i + 16) * ch` bytes and each
            // destination pointer has `width >= i + 16` writable bytes.
            unsafe {
                let s_off = sp.add(i * ch);
                let (r0, g0, b0, a0) = load8_f32_u8::<KIND>(s_off, p);
                let (r1, g1, b1, a1) = load8_f32_u8::<KIND>(s_off.add(8 * ch), p);
                if KIND != 2 && FUSE_A {
                    store16_u8_ptr(ap.add(i), a0, a1);
                }
                let ((y0, u0, v0), (y1, u1, v1)) = yuv_coeffs_pair(p, r0, g0, b0, r1, g1, b1);
                let qy0 = quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half);
                let qy1 = quant_i32(y1, p.v_range_y, p.v_bias_y, p.v_half);
                store16_u8_ptr(yp.add(i), qy0, qy1);
                let qu0 = quant_i32(u0, p.v_range_uv, p.v_bias_uv, p.v_half);
                let qu1 = quant_i32(u1, p.v_range_uv, p.v_bias_uv, p.v_half);
                store16_u8_ptr(up.add(i), qu0, qu1);
                let qv0 = quant_i32(v0, p.v_range_uv, p.v_bias_uv, p.v_half);
                let qv1 = quant_i32(v1, p.v_range_uv, p.v_bias_uv, p.v_half);
                store16_u8_ptr(vp.add(i), qv0, qv1);
            }
            i += 16;
        }
        if i + 8 <= width {
            // SAFETY: Entry bounds checks guarantee `i + 8 <= width`.
            unsafe {
                let (r0, g0, b0, a0) = load8_f32_u8::<KIND>(sp.add(i * ch), p);
                if KIND != 2 && FUSE_A {
                    store8_u8_ptr(ap.add(i), a0);
                }
                let (y0, u0, v0) = yuv_one_mode::<MODE_COEFFS>(p, r0, g0, b0);
                store8_u8_ptr(yp.add(i), quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half));
                store8_u8_ptr(
                    up.add(i),
                    quant_i32(u0, p.v_range_uv, p.v_bias_uv, p.v_half),
                );
                store8_u8_ptr(
                    vp.add(i),
                    quant_i32(v0, p.v_range_uv, p.v_bias_uv, p.v_half),
                );
            }
            i += 8;
        }
        for x in i..width {
            // SAFETY: `x < width` and entry bounds checks guarantee valid reads and writes.
            unsafe {
                let (yuv, a) = scalar_px_spec::<MODE_COEFFS, true>(sp, std::ptr::null(), x, fmt, p);
                *yp.add(x) = to_unorm8!(p.bias_y, p.range_y, yuv.0);
                *up.add(x) = to_unorm8!(p.bias_uv, p.range_uv, yuv.1);
                *vp.add(x) = to_unorm8!(p.bias_uv, p.range_uv, yuv.2);
                if KIND != 2 && FUSE_A {
                    *ap.add(x) = a as u8;
                }
            }
        }
        Ok(())
    }

    /// General AVX2 `Yuv444` row kernel covering all bit depths (`8, 10, 12, 16`) and all modes.
    ///
    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_444_general_spec<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        src8: &[u8],
        src16: &[u16],
        dst_y8: &mut [MaybeUninit<u8>],
        dst_u8: &mut [MaybeUninit<u8>],
        dst_v8: &mut [MaybeUninit<u8>],
        dst_a8: &mut [MaybeUninit<u8>],
        dst_y16: &mut [MaybeUninit<u16>],
        dst_u16: &mut [MaybeUninit<u16>],
        dst_v16: &mut [MaybeUninit<u16>],
        dst_a16: &mut [MaybeUninit<u16>],
        width: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        if ch != if KIND == 2 { 3 } else { 4 } {
            return AvifError::no_content();
        }
        let src_req = width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        let (s8p, s16p) = if RGB_8 {
            if src8.len() < src_req {
                return AvifError::no_content();
            }
            (src8.as_ptr(), std::ptr::null())
        } else {
            if src16.len() < src_req {
                return AvifError::no_content();
            }
            (std::ptr::null(), src16.as_ptr())
        };
        let (y8p, u8p, v8p, a8p, y16p, u16p, v16p, a16p) = if IMG_8 {
            if dst_y8.len() < width || dst_u8.len() < width || dst_v8.len() < width {
                return AvifError::no_content();
            }
            let ap = if KIND != 2 && HAS_A {
                if dst_a8.len() < width {
                    return AvifError::no_content();
                }
                dst_a8.as_mut_ptr().cast::<u8>()
            } else {
                std::ptr::null_mut()
            };
            (
                dst_y8.as_mut_ptr().cast::<u8>(),
                dst_u8.as_mut_ptr().cast::<u8>(),
                dst_v8.as_mut_ptr().cast::<u8>(),
                ap,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } else {
            if dst_y16.len() < width || dst_u16.len() < width || dst_v16.len() < width {
                return AvifError::no_content();
            }
            let ap = if KIND != 2 && HAS_A {
                if dst_a16.len() < width {
                    return AvifError::no_content();
                }
                dst_a16.as_mut_ptr().cast::<u16>()
            } else {
                std::ptr::null_mut()
            };
            (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                dst_y16.as_mut_ptr().cast::<u16>(),
                dst_u16.as_mut_ptr().cast::<u16>(),
                dst_v16.as_mut_ptr().cast::<u16>(),
                ap,
            )
        };

        let mut i = 0usize;
        while i + 16 <= width {
            // SAFETY: Loop guard `i + 16 <= width` and entry length checks guarantee readable
            // input and writable output slots for `[i..i + 16)`.
            unsafe {
                let ((r0, g0, b0, a0), (r1, g1, b1, a1)) = if !RGB_8 {
                    let sp = s16p.add(i * ch);
                    (
                        load8_f32_u16::<KIND>(sp, p),
                        load8_f32_u16::<KIND>(sp.add(8 * ch), p),
                    )
                } else {
                    let sp = s8p.add(i * ch);
                    (
                        load8_f32_u8::<KIND>(sp, p),
                        load8_f32_u8::<KIND>(sp.add(8 * ch), p),
                    )
                };
                let ((y0, u0, v0), (y1, u1, v1)) =
                    yuv_pair_mode::<MODE_TAG>(p, r0, g0, b0, r1, g1, b1);
                let qy0 = quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half);
                let qy1 = quant_i32(y1, p.v_range_y, p.v_bias_y, p.v_half);
                let qu0 = quant_i32(u0, p.v_range_uv, p.v_bias_uv, p.v_half);
                let qu1 = quant_i32(u1, p.v_range_uv, p.v_bias_uv, p.v_half);
                let qv0 = quant_i32(v0, p.v_range_uv, p.v_bias_uv, p.v_half);
                let qv1 = quant_i32(v1, p.v_range_uv, p.v_bias_uv, p.v_half);
                if !IMG_8 {
                    store16_u16_ptr(y16p.add(i), qy0, qy1, p.v_max_u16);
                    store16_u16_ptr(u16p.add(i), qu0, qu1, p.v_max_u16);
                    store16_u16_ptr(v16p.add(i), qv0, qv1, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u16(a16p.add(i), a0, a1, p);
                    }
                } else {
                    store16_u8_ptr(y8p.add(i), qy0, qy1);
                    store16_u8_ptr(u8p.add(i), qu0, qu1);
                    store16_u8_ptr(v8p.add(i), qv0, qv1);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u8(a8p.add(i), a0, a1, p);
                    }
                }
            }
            i += 16;
        }
        if i + 8 <= width {
            // SAFETY: `i + 8 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (r0, g0, b0, a0) = if !RGB_8 {
                    load8_f32_u16::<KIND>(s16p.add(i * ch), p)
                } else {
                    load8_f32_u8::<KIND>(s8p.add(i * ch), p)
                };
                let (y0, u0, v0) = yuv_one_mode::<MODE_TAG>(p, r0, g0, b0);
                let qy = quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half);
                let qu = quant_i32(u0, p.v_range_uv, p.v_bias_uv, p.v_half);
                let qv = quant_i32(v0, p.v_range_uv, p.v_bias_uv, p.v_half);
                if !IMG_8 {
                    store8_u16_ptr(y16p.add(i), qy, p.v_max_u16);
                    store8_u16_ptr(u16p.add(i), qu, p.v_max_u16);
                    store8_u16_ptr(v16p.add(i), qv, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u16(a16p.add(i), a0, p);
                    }
                } else {
                    store8_u8_ptr(y8p.add(i), qy);
                    store8_u8_ptr(u8p.add(i), qu);
                    store8_u8_ptr(v8p.add(i), qv);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u8(a8p.add(i), a0, p);
                    }
                }
            }
            i += 8;
        }
        for x in i..width {
            // SAFETY: `x < width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (yuv, a) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p, s16p, x, fmt, p);
                if !IMG_8 {
                    *y16p.add(x) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, yuv.0);
                    *u16p.add(x) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, yuv.1);
                    *v16p.add(x) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, yuv.2);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u16(a16p.add(x), a, p);
                    }
                } else {
                    *y8p.add(x) = to_unorm8!(p.bias_y, p.range_y, yuv.0);
                    *u8p.add(x) = to_unorm8!(p.bias_uv, p.range_uv, yuv.1);
                    *v8p.add(x) = to_unorm8!(p.bias_uv, p.range_uv, yuv.2);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u8(a8p.add(x), a, p);
                    }
                }
            }
        }
        Ok(())
    }

    /// AVX2 `Yuv400` row kernel covering all bit depths (`8, 10, 12, 16`), modes, and fused alpha.
    ///
    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_400_spec<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        src8: &[u8],
        src16: &[u16],
        dst_y8: &mut [MaybeUninit<u8>],
        dst_a8: &mut [MaybeUninit<u8>],
        dst_y16: &mut [MaybeUninit<u16>],
        dst_a16: &mut [MaybeUninit<u16>],
        width: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        if ch != if KIND == 2 { 3 } else { 4 } {
            return AvifError::no_content();
        }
        let src_req = width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        let (s8p, s16p) = if RGB_8 {
            if src8.len() < src_req {
                return AvifError::no_content();
            }
            (src8.as_ptr(), std::ptr::null())
        } else {
            if src16.len() < src_req {
                return AvifError::no_content();
            }
            (std::ptr::null(), src16.as_ptr())
        };
        let (y8p, a8p, y16p, a16p) = if IMG_8 {
            if dst_y8.len() < width {
                return AvifError::no_content();
            }
            let ap = if KIND != 2 && HAS_A {
                if dst_a8.len() < width {
                    return AvifError::no_content();
                }
                dst_a8.as_mut_ptr().cast::<u8>()
            } else {
                std::ptr::null_mut()
            };
            (
                dst_y8.as_mut_ptr().cast::<u8>(),
                ap,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } else {
            if dst_y16.len() < width {
                return AvifError::no_content();
            }
            let ap = if KIND != 2 && HAS_A {
                if dst_a16.len() < width {
                    return AvifError::no_content();
                }
                dst_a16.as_mut_ptr().cast::<u16>()
            } else {
                std::ptr::null_mut()
            };
            (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                dst_y16.as_mut_ptr().cast::<u16>(),
                ap,
            )
        };

        let mut i = 0usize;
        while i + 16 <= width {
            // SAFETY: `i + 16 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let ((r0, g0, b0, a0), (r1, g1, b1, a1)) = if !RGB_8 {
                    let sp = s16p.add(i * ch);
                    (
                        load8_f32_u16::<KIND>(sp, p),
                        load8_f32_u16::<KIND>(sp.add(8 * ch), p),
                    )
                } else {
                    let sp = s8p.add(i * ch);
                    (
                        load8_f32_u8::<KIND>(sp, p),
                        load8_f32_u8::<KIND>(sp.add(8 * ch), p),
                    )
                };
                let y0 = luma_one_mode::<MODE_TAG>(p, r0, g0, b0);
                let y1 = luma_one_mode::<MODE_TAG>(p, r1, g1, b1);
                let qy0 = quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half);
                let qy1 = quant_i32(y1, p.v_range_y, p.v_bias_y, p.v_half);
                if !IMG_8 {
                    store16_u16_ptr(y16p.add(i), qy0, qy1, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u16(a16p.add(i), a0, a1, p);
                    }
                } else {
                    store16_u8_ptr(y8p.add(i), qy0, qy1);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u8(a8p.add(i), a0, a1, p);
                    }
                }
            }
            i += 16;
        }
        if i + 8 <= width {
            // SAFETY: `i + 8 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (r0, g0, b0, a0) = if !RGB_8 {
                    load8_f32_u16::<KIND>(s16p.add(i * ch), p)
                } else {
                    load8_f32_u8::<KIND>(s8p.add(i * ch), p)
                };
                let qy = quant_i32(
                    luma_one_mode::<MODE_TAG>(p, r0, g0, b0),
                    p.v_range_y,
                    p.v_bias_y,
                    p.v_half,
                );
                if !IMG_8 {
                    store8_u16_ptr(y16p.add(i), qy, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u16(a16p.add(i), a0, p);
                    }
                } else {
                    store8_u8_ptr(y8p.add(i), qy);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u8(a8p.add(i), a0, p);
                    }
                }
            }
            i += 8;
        }
        for x in i..width {
            // SAFETY: `x < width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (yuv, a) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p, s16p, x, fmt, p);
                if !IMG_8 {
                    *y16p.add(x) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, yuv.0);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u16(a16p.add(x), a, p);
                    }
                } else {
                    *y8p.add(x) = to_unorm8!(p.bias_y, p.range_y, yuv.0);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u8(a8p.add(x), a, p);
                    }
                }
            }
        }
        Ok(())
    }

    /// AVX2 `Yuv422` row kernel (also handles the odd final row of `Yuv420`).
    ///
    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_422_spec<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        src8: &[u8],
        src16: &[u16],
        dst_y8: &mut [MaybeUninit<u8>],
        dst_u8: &mut [MaybeUninit<u8>],
        dst_v8: &mut [MaybeUninit<u8>],
        dst_a8: &mut [MaybeUninit<u8>],
        dst_y16: &mut [MaybeUninit<u16>],
        dst_u16: &mut [MaybeUninit<u16>],
        dst_v16: &mut [MaybeUninit<u16>],
        dst_a16: &mut [MaybeUninit<u16>],
        width: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        if ch != if KIND == 2 { 3 } else { 4 } {
            return AvifError::no_content();
        }
        let uv_width = width.div_ceil(2);
        let src_req = width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        let (s8p, s16p) = if RGB_8 {
            if src8.len() < src_req {
                return AvifError::no_content();
            }
            (src8.as_ptr(), std::ptr::null())
        } else {
            if src16.len() < src_req {
                return AvifError::no_content();
            }
            (std::ptr::null(), src16.as_ptr())
        };
        let (y8p, u8p, v8p, a8p, y16p, u16p, v16p, a16p) = if IMG_8 {
            if dst_y8.len() < width || dst_u8.len() < uv_width || dst_v8.len() < uv_width {
                return AvifError::no_content();
            }
            let ap = if KIND != 2 && HAS_A {
                if dst_a8.len() < width {
                    return AvifError::no_content();
                }
                dst_a8.as_mut_ptr().cast::<u8>()
            } else {
                std::ptr::null_mut()
            };
            (
                dst_y8.as_mut_ptr().cast::<u8>(),
                dst_u8.as_mut_ptr().cast::<u8>(),
                dst_v8.as_mut_ptr().cast::<u8>(),
                ap,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } else {
            if dst_y16.len() < width || dst_u16.len() < uv_width || dst_v16.len() < uv_width {
                return AvifError::no_content();
            }
            let ap = if KIND != 2 && HAS_A {
                if dst_a16.len() < width {
                    return AvifError::no_content();
                }
                dst_a16.as_mut_ptr().cast::<u16>()
            } else {
                std::ptr::null_mut()
            };
            (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                dst_y16.as_mut_ptr().cast::<u16>(),
                dst_u16.as_mut_ptr().cast::<u16>(),
                dst_v16.as_mut_ptr().cast::<u16>(),
                ap,
            )
        };

        let mut i = 0usize;
        while i + 16 <= width {
            let uv_i = i >> 1;
            // SAFETY: `i + 16 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let ((r0, g0, b0, a0), (r1, g1, b1, a1)) = if !RGB_8 {
                    let sp = s16p.add(i * ch);
                    (
                        load8_f32_u16::<KIND>(sp, p),
                        load8_f32_u16::<KIND>(sp.add(8 * ch), p),
                    )
                } else {
                    let sp = s8p.add(i * ch);
                    (
                        load8_f32_u8::<KIND>(sp, p),
                        load8_f32_u8::<KIND>(sp.add(8 * ch), p),
                    )
                };
                let ((y0, u0, v0), (y1, u1, v1)) =
                    yuv_pair_mode::<MODE_TAG>(p, r0, g0, b0, r1, g1, b1);
                let qy0 = quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half);
                let qy1 = quant_i32(y1, p.v_range_y, p.v_bias_y, p.v_half);
                let qu = quant_avg16_422(u0, u1, p);
                let qv = quant_avg16_422(v0, v1, p);
                if !IMG_8 {
                    store16_u16_ptr(y16p.add(i), qy0, qy1, p.v_max_u16);
                    store8_u16_ptr(u16p.add(uv_i), qu, p.v_max_u16);
                    store8_u16_ptr(v16p.add(uv_i), qv, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u16(a16p.add(i), a0, a1, p);
                    }
                } else {
                    store16_u8_ptr(y8p.add(i), qy0, qy1);
                    store8_u8_ptr(u8p.add(uv_i), qu);
                    store8_u8_ptr(v8p.add(uv_i), qv);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u8(a8p.add(i), a0, a1, p);
                    }
                }
            }
            i += 16;
        }
        if i + 8 <= width {
            let uv_i = i >> 1;
            // SAFETY: `i + 8 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (r0, g0, b0, a0) = if !RGB_8 {
                    load8_f32_u16::<KIND>(s16p.add(i * ch), p)
                } else {
                    load8_f32_u8::<KIND>(s8p.add(i * ch), p)
                };
                let (y0, u0, v0) = yuv_one_mode::<MODE_TAG>(p, r0, g0, b0);
                let qy = quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half);
                let qu4 = quant_avg8_422(u0, p);
                let qv4 = quant_avg8_422(v0, p);
                if !IMG_8 {
                    store8_u16_ptr(y16p.add(i), qy, p.v_max_u16);
                    store4_u16_ptr(u16p.add(uv_i), qu4, p.v_max_u16);
                    store4_u16_ptr(v16p.add(uv_i), qv4, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u16(a16p.add(i), a0, p);
                    }
                } else {
                    store8_u8_ptr(y8p.add(i), qy);
                    store4_u8_ptr(u8p.add(uv_i), qu4);
                    store4_u8_ptr(v8p.add(uv_i), qv4);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u8(a8p.add(i), a0, p);
                    }
                }
            }
            i += 8;
        }
        while i + 2 <= width {
            // SAFETY: `i + 2 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (p0, a0) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p, s16p, i, fmt, p);
                let (p1, a1) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p, s16p, i + 1, fmt, p);
                let avg_u = (p0.1 + p1.1) / 2.0;
                let avg_v = (p0.2 + p1.2) / 2.0;
                let uv_i = i >> 1;
                if !IMG_8 {
                    *y16p.add(i) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p0.0);
                    *y16p.add(i + 1) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p1.0);
                    *u16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, avg_u);
                    *v16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, avg_v);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u16(a16p.add(i), a0, p);
                        store1_fused_alpha_u16(a16p.add(i + 1), a1, p);
                    }
                } else {
                    *y8p.add(i) = to_unorm8!(p.bias_y, p.range_y, p0.0);
                    *y8p.add(i + 1) = to_unorm8!(p.bias_y, p.range_y, p1.0);
                    *u8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, avg_u);
                    *v8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, avg_v);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u8(a8p.add(i), a0, p);
                        store1_fused_alpha_u8(a8p.add(i + 1), a1, p);
                    }
                }
            }
            i += 2;
        }
        if i < width {
            // SAFETY: `i < width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (p0, a0) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p, s16p, i, fmt, p);
                let uv_i = i >> 1;
                if !IMG_8 {
                    *y16p.add(i) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p0.0);
                    *u16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, p0.1);
                    *v16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, p0.2);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u16(a16p.add(i), a0, p);
                    }
                } else {
                    *y8p.add(i) = to_unorm8!(p.bias_y, p.range_y, p0.0);
                    *u8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, p0.1);
                    *v8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, p0.2);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u8(a8p.add(i), a0, p);
                    }
                }
            }
        }
        Ok(())
    }

    /// AVX2 `Yuv420` paired-row kernel with exact left-associative 2x2 chroma box averaging.
    ///
    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn two_rows_420_spec<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        src8_0: &[u8],
        src8_1: &[u8],
        src16_0: &[u16],
        src16_1: &[u16],
        dst_y8_0: &mut [MaybeUninit<u8>],
        dst_y8_1: &mut [MaybeUninit<u8>],
        dst_u8: &mut [MaybeUninit<u8>],
        dst_v8: &mut [MaybeUninit<u8>],
        dst_a8_0: &mut [MaybeUninit<u8>],
        dst_a8_1: &mut [MaybeUninit<u8>],
        dst_y16_0: &mut [MaybeUninit<u16>],
        dst_y16_1: &mut [MaybeUninit<u16>],
        dst_u16: &mut [MaybeUninit<u16>],
        dst_v16: &mut [MaybeUninit<u16>],
        dst_a16_0: &mut [MaybeUninit<u16>],
        dst_a16_1: &mut [MaybeUninit<u16>],
        width: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        if ch != if KIND == 2 { 3 } else { 4 } {
            return AvifError::no_content();
        }
        let uv_width = width.div_ceil(2);
        let src_req = width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        let (s8p0, s8p1, s16p0, s16p1) = if RGB_8 {
            if src8_0.len() < src_req || src8_1.len() < src_req {
                return AvifError::no_content();
            }
            (
                src8_0.as_ptr(),
                src8_1.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        } else {
            if src16_0.len() < src_req || src16_1.len() < src_req {
                return AvifError::no_content();
            }
            (
                std::ptr::null(),
                std::ptr::null(),
                src16_0.as_ptr(),
                src16_1.as_ptr(),
            )
        };
        let (y8p0, y8p1, u8p, v8p, a8p0, a8p1, y16p0, y16p1, u16p, v16p, a16p0, a16p1) = if IMG_8 {
            if dst_y8_0.len() < width
                || dst_y8_1.len() < width
                || dst_u8.len() < uv_width
                || dst_v8.len() < uv_width
            {
                return AvifError::no_content();
            }
            let (ap0, ap1) = if KIND != 2 && HAS_A {
                if dst_a8_0.len() < width || dst_a8_1.len() < width {
                    return AvifError::no_content();
                }
                (
                    dst_a8_0.as_mut_ptr().cast::<u8>(),
                    dst_a8_1.as_mut_ptr().cast::<u8>(),
                )
            } else {
                (std::ptr::null_mut(), std::ptr::null_mut())
            };
            (
                dst_y8_0.as_mut_ptr().cast::<u8>(),
                dst_y8_1.as_mut_ptr().cast::<u8>(),
                dst_u8.as_mut_ptr().cast::<u8>(),
                dst_v8.as_mut_ptr().cast::<u8>(),
                ap0,
                ap1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } else {
            if dst_y16_0.len() < width
                || dst_y16_1.len() < width
                || dst_u16.len() < uv_width
                || dst_v16.len() < uv_width
            {
                return AvifError::no_content();
            }
            let (ap0, ap1) = if KIND != 2 && HAS_A {
                if dst_a16_0.len() < width || dst_a16_1.len() < width {
                    return AvifError::no_content();
                }
                (
                    dst_a16_0.as_mut_ptr().cast::<u16>(),
                    dst_a16_1.as_mut_ptr().cast::<u16>(),
                )
            } else {
                (std::ptr::null_mut(), std::ptr::null_mut())
            };
            (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                dst_y16_0.as_mut_ptr().cast::<u16>(),
                dst_y16_1.as_mut_ptr().cast::<u16>(),
                dst_u16.as_mut_ptr().cast::<u16>(),
                dst_v16.as_mut_ptr().cast::<u16>(),
                ap0,
                ap1,
            )
        };

        let mut i = 0usize;
        while i + 16 <= width {
            let uv_i = i >> 1;
            // SAFETY: `i + 16 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (row0, row1) = if !RGB_8 {
                    let sp0 = s16p0.add(i * ch);
                    let sp1 = s16p1.add(i * ch);
                    (
                        (
                            load8_f32_u16::<KIND>(sp0, p),
                            load8_f32_u16::<KIND>(sp0.add(8 * ch), p),
                        ),
                        (
                            load8_f32_u16::<KIND>(sp1, p),
                            load8_f32_u16::<KIND>(sp1.add(8 * ch), p),
                        ),
                    )
                } else {
                    let sp0 = s8p0.add(i * ch);
                    let sp1 = s8p1.add(i * ch);
                    (
                        (
                            load8_f32_u8::<KIND>(sp0, p),
                            load8_f32_u8::<KIND>(sp0.add(8 * ch), p),
                        ),
                        (
                            load8_f32_u8::<KIND>(sp1, p),
                            load8_f32_u8::<KIND>(sp1.add(8 * ch), p),
                        ),
                    )
                };
                let ((r00, g00, b00, a00), (r01, g01, b01, a01)) = row0;
                let ((r10, g10, b10, a10), (r11, g11, b11, a11)) = row1;
                let ((y00, u00, v00), (y01, u01, v01)) =
                    yuv_pair_mode::<MODE_TAG>(p, r00, g00, b00, r01, g01, b01);
                let ((y10, u10, v10), (y11, u11, v11)) =
                    yuv_pair_mode::<MODE_TAG>(p, r10, g10, b10, r11, g11, b11);
                let qy00 = quant_i32(y00, p.v_range_y, p.v_bias_y, p.v_half);
                let qy01 = quant_i32(y01, p.v_range_y, p.v_bias_y, p.v_half);
                let qy10 = quant_i32(y10, p.v_range_y, p.v_bias_y, p.v_half);
                let qy11 = quant_i32(y11, p.v_range_y, p.v_bias_y, p.v_half);
                let qu = quant_avg16_420(u00, u01, u10, u11, p);
                let qv = quant_avg16_420(v00, v01, v10, v11, p);
                if !IMG_8 {
                    store16_u16_ptr(y16p0.add(i), qy00, qy01, p.v_max_u16);
                    store16_u16_ptr(y16p1.add(i), qy10, qy11, p.v_max_u16);
                    store8_u16_ptr(u16p.add(uv_i), qu, p.v_max_u16);
                    store8_u16_ptr(v16p.add(uv_i), qv, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u16(a16p0.add(i), a00, a01, p);
                        store16_fused_alpha_u16(a16p1.add(i), a10, a11, p);
                    }
                } else {
                    store16_u8_ptr(y8p0.add(i), qy00, qy01);
                    store16_u8_ptr(y8p1.add(i), qy10, qy11);
                    store8_u8_ptr(u8p.add(uv_i), qu);
                    store8_u8_ptr(v8p.add(uv_i), qv);
                    if KIND != 2 && HAS_A {
                        store16_fused_alpha_u8(a8p0.add(i), a00, a01, p);
                        store16_fused_alpha_u8(a8p1.add(i), a10, a11, p);
                    }
                }
            }
            i += 16;
        }
        if i + 8 <= width {
            let uv_i = i >> 1;
            // SAFETY: `i + 8 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let ((r0, g0, b0, a0), (r1, g1, b1, a1)) = if !RGB_8 {
                    (
                        load8_f32_u16::<KIND>(s16p0.add(i * ch), p),
                        load8_f32_u16::<KIND>(s16p1.add(i * ch), p),
                    )
                } else {
                    (
                        load8_f32_u8::<KIND>(s8p0.add(i * ch), p),
                        load8_f32_u8::<KIND>(s8p1.add(i * ch), p),
                    )
                };
                let ((y0, u0, v0), (y1, u1, v1)) =
                    yuv_pair_mode::<MODE_TAG>(p, r0, g0, b0, r1, g1, b1);
                let qy0 = quant_i32(y0, p.v_range_y, p.v_bias_y, p.v_half);
                let qy1 = quant_i32(y1, p.v_range_y, p.v_bias_y, p.v_half);
                let qu4 = quant_avg8_420(u0, u1, p);
                let qv4 = quant_avg8_420(v0, v1, p);
                if !IMG_8 {
                    store8_u16_ptr(y16p0.add(i), qy0, p.v_max_u16);
                    store8_u16_ptr(y16p1.add(i), qy1, p.v_max_u16);
                    store4_u16_ptr(u16p.add(uv_i), qu4, p.v_max_u16);
                    store4_u16_ptr(v16p.add(uv_i), qv4, p.v_max_u16);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u16(a16p0.add(i), a0, p);
                        store8_fused_alpha_u16(a16p1.add(i), a1, p);
                    }
                } else {
                    store8_u8_ptr(y8p0.add(i), qy0);
                    store8_u8_ptr(y8p1.add(i), qy1);
                    store4_u8_ptr(u8p.add(uv_i), qu4);
                    store4_u8_ptr(v8p.add(uv_i), qv4);
                    if KIND != 2 && HAS_A {
                        store8_fused_alpha_u8(a8p0.add(i), a0, p);
                        store8_fused_alpha_u8(a8p1.add(i), a1, p);
                    }
                }
            }
            i += 8;
        }
        while i + 2 <= width {
            // SAFETY: `i + 2 <= width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (p00, a00) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p0, s16p0, i, fmt, p);
                let (p01, a01) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p0, s16p0, i + 1, fmt, p);
                let (p10, a10) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p1, s16p1, i, fmt, p);
                let (p11, a11) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p1, s16p1, i + 1, fmt, p);
                let avg_u = (p00.1 + p01.1 + p10.1 + p11.1) / 4.0;
                let avg_v = (p00.2 + p01.2 + p10.2 + p11.2) / 4.0;
                let uv_i = i >> 1;
                if !IMG_8 {
                    *y16p0.add(i) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p00.0);
                    *y16p0.add(i + 1) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p01.0);
                    *y16p1.add(i) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p10.0);
                    *y16p1.add(i + 1) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p11.0);
                    *u16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, avg_u);
                    *v16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, avg_v);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u16(a16p0.add(i), a00, p);
                        store1_fused_alpha_u16(a16p0.add(i + 1), a01, p);
                        store1_fused_alpha_u16(a16p1.add(i), a10, p);
                        store1_fused_alpha_u16(a16p1.add(i + 1), a11, p);
                    }
                } else {
                    *y8p0.add(i) = to_unorm8!(p.bias_y, p.range_y, p00.0);
                    *y8p0.add(i + 1) = to_unorm8!(p.bias_y, p.range_y, p01.0);
                    *y8p1.add(i) = to_unorm8!(p.bias_y, p.range_y, p10.0);
                    *y8p1.add(i + 1) = to_unorm8!(p.bias_y, p.range_y, p11.0);
                    *u8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, avg_u);
                    *v8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, avg_v);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u8(a8p0.add(i), a00, p);
                        store1_fused_alpha_u8(a8p0.add(i + 1), a01, p);
                        store1_fused_alpha_u8(a8p1.add(i), a10, p);
                        store1_fused_alpha_u8(a8p1.add(i + 1), a11, p);
                    }
                }
            }
            i += 2;
        }
        if i < width {
            // SAFETY: `i < width` and entry length checks guarantee valid reads and writes.
            unsafe {
                let (p0, a0) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p0, s16p0, i, fmt, p);
                let (p1, a1) = scalar_px_spec::<MODE_TAG, RGB_8>(s8p1, s16p1, i, fmt, p);
                let avg_u = (p0.1 + p1.1) / 2.0;
                let avg_v = (p0.2 + p1.2) / 2.0;
                let uv_i = i >> 1;
                if !IMG_8 {
                    *y16p0.add(i) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p0.0);
                    *y16p1.add(i) = to_unorm16!(p.bias_y, p.range_y, p.yuv_max, p1.0);
                    *u16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, avg_u);
                    *v16p.add(uv_i) = to_unorm16!(p.bias_uv, p.range_uv, p.yuv_max, avg_v);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u16(a16p0.add(i), a0, p);
                        store1_fused_alpha_u16(a16p1.add(i), a1, p);
                    }
                } else {
                    *y8p0.add(i) = to_unorm8!(p.bias_y, p.range_y, p0.0);
                    *y8p1.add(i) = to_unorm8!(p.bias_y, p.range_y, p1.0);
                    *u8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, avg_u);
                    *v8p.add(uv_i) = to_unorm8!(p.bias_uv, p.range_uv, avg_v);
                    if KIND != 2 && HAS_A {
                        store1_fused_alpha_u8(a8p0.add(i), a0, p);
                        store1_fused_alpha_u8(a8p1.add(i), a1, p);
                    }
                }
            }
        }
        Ok(())
    }

    macro_rules! dispatch_spec {
        (
            $fmt:expr,
            $mode_tag:expr,
            $rgb8:expr,
            $img8:expr,
            $has_a:expr,
            $fn:ident ( $($arg:expr),* $(,)? )
        ) => {{
            macro_rules! with_fmt {
                ($mt:literal, $r8:literal, $i8:literal) => {
                    match ($fmt, $has_a) {
                        (RgbFmt::Rgba, true) => $fn::<0, $mt, $r8, $i8, true>($($arg),*),
                        (RgbFmt::Rgba, false) => $fn::<0, $mt, $r8, $i8, false>($($arg),*),
                        (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, true) => {
                            $fn::<1, $mt, $r8, $i8, true>($($arg),*)
                        }
                        (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, false) => {
                            $fn::<1, $mt, $r8, $i8, false>($($arg),*)
                        }
                        (RgbFmt::Rgb | RgbFmt::Bgr, _) => {
                            $fn::<2, $mt, $r8, $i8, false>($($arg),*)
                        }
                    }
                };
            }
            macro_rules! with_depths {
                ($mt:literal) => {
                    match ($rgb8, $img8) {
                        (true, true) => with_fmt!($mt, true, true),
                        (true, false) => with_fmt!($mt, true, false),
                        (false, true) => with_fmt!($mt, false, true),
                        (false, false) => with_fmt!($mt, false, false),
                    }
                };
            }
            match $mode_tag {
                MODE_COEFFS => with_depths!(0),
                MODE_IDENTITY => with_depths!(1),
                MODE_YCGCO => with_depths!(2),
                _ => with_depths!(3),
            }
        }};
    }

    #[derive(Clone, Copy)]
    struct YuvStripe {
        src8: *const u8,
        src16: *const u16,
        src_stride_elems: usize,
        dst_y8: *mut MaybeUninit<u8>,
        dst_u8: *mut MaybeUninit<u8>,
        dst_v8: *mut MaybeUninit<u8>,
        dst_a8: *mut MaybeUninit<u8>,
        dst_y16: *mut MaybeUninit<u16>,
        dst_u16: *mut MaybeUninit<u16>,
        dst_v16: *mut MaybeUninit<u16>,
        dst_a16: *mut MaybeUninit<u16>,
        y_stride_elems: usize,
        u_stride_elems: usize,
        v_stride_elems: usize,
        a_stride_elems: usize,
        width: usize,
        row_count: usize,
    }

    // SAFETY: Each `YuvStripe` holds read-only source pointers validated across the input image
    // and writable destination pointers covering a row range disjoint from every other stripe in
    // the same scoped conversion.
    unsafe impl Send for YuvStripe {}

    /// # Safety
    /// Caller must ensure `avx2` support and that `stripe` pointers cover
    /// `(0..eff_rows) x (0..eff_width)` for read (source) and exclusive write (destination).
    #[target_feature(enable = "avx2")]
    unsafe fn run_444_identity_u8_stripe<const KIND: u8, const FUSE_A: bool>(
        stripe: &YuvStripe,
        eff_width: usize,
        eff_rows: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        let src_req = eff_width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        for local_j in 0..eff_rows {
            // SAFETY: Caller guarantees stripe bounds; `local_j < eff_rows`.
            unsafe {
                let s = std::slice::from_raw_parts(
                    stripe.src8.add(local_j * stripe.src_stride_elems),
                    src_req,
                );
                let dy = std::slice::from_raw_parts_mut(
                    stripe.dst_y8.add(local_j * stripe.y_stride_elems),
                    eff_width,
                );
                let du = std::slice::from_raw_parts_mut(
                    stripe.dst_u8.add(local_j * stripe.u_stride_elems),
                    eff_width,
                );
                let dv = std::slice::from_raw_parts_mut(
                    stripe.dst_v8.add(local_j * stripe.v_stride_elems),
                    eff_width,
                );
                let da = if KIND != 2 && FUSE_A {
                    Some(std::slice::from_raw_parts_mut(
                        stripe.dst_a8.add(local_j * stripe.a_stride_elems),
                        eff_width,
                    ))
                } else {
                    None
                };
                row_444_identity_u8::<KIND, FUSE_A>(s, dy, du, dv, da, eff_width, fmt, p)?;
            }
        }
        Ok(())
    }

    /// # Safety
    /// Caller must ensure `avx2` support and that `stripe` pointers cover
    /// `(0..eff_rows) x (0..eff_width)` for read (source) and exclusive write (destination).
    #[target_feature(enable = "avx2")]
    unsafe fn run_444_u8_coeffs_stripe<const KIND: u8, const FUSE_A: bool>(
        stripe: &YuvStripe,
        eff_width: usize,
        eff_rows: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        let src_req = eff_width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        for local_j in 0..eff_rows {
            // SAFETY: Caller guarantees stripe bounds; `local_j < eff_rows`.
            unsafe {
                let s = std::slice::from_raw_parts(
                    stripe.src8.add(local_j * stripe.src_stride_elems),
                    src_req,
                );
                let dy = std::slice::from_raw_parts_mut(
                    stripe.dst_y8.add(local_j * stripe.y_stride_elems),
                    eff_width,
                );
                let du = std::slice::from_raw_parts_mut(
                    stripe.dst_u8.add(local_j * stripe.u_stride_elems),
                    eff_width,
                );
                let dv = std::slice::from_raw_parts_mut(
                    stripe.dst_v8.add(local_j * stripe.v_stride_elems),
                    eff_width,
                );
                let da = if KIND != 2 && FUSE_A {
                    Some(std::slice::from_raw_parts_mut(
                        stripe.dst_a8.add(local_j * stripe.a_stride_elems),
                        eff_width,
                    ))
                } else {
                    None
                };
                row_444_u8_coeffs::<KIND, FUSE_A>(s, dy, du, dv, da, eff_width, fmt, p)?;
            }
        }
        Ok(())
    }

    /// # Safety
    /// Caller must ensure `avx2` support and that `stripe` pointers cover
    /// `(0..eff_rows) x (0..eff_width)` for read (source) and exclusive write (destination).
    #[target_feature(enable = "avx2")]
    unsafe fn run_444_general_stripe<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        stripe: &YuvStripe,
        eff_width: usize,
        eff_rows: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        let src_req = eff_width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        for local_j in 0..eff_rows {
            // SAFETY: Caller guarantees stripe bounds; `local_j < eff_rows`.
            unsafe {
                let (s8, s16): (&[u8], &[u16]) = if !RGB_8 {
                    (
                        &[],
                        std::slice::from_raw_parts(
                            stripe.src16.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                    )
                } else {
                    (
                        std::slice::from_raw_parts(
                            stripe.src8.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                        &[],
                    )
                };
                if !IMG_8 {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y16.add(local_j * stripe.y_stride_elems),
                        eff_width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u16.add(local_j * stripe.u_stride_elems),
                        eff_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v16.add(local_j * stripe.v_stride_elems),
                        eff_width,
                    );
                    let da: &mut [MaybeUninit<u16>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a16.add(local_j * stripe.a_stride_elems),
                            eff_width,
                        )
                    } else {
                        &mut []
                    };
                    row_444_general_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        dy,
                        du,
                        dv,
                        da,
                        eff_width,
                        fmt,
                        p,
                    )?;
                } else {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y8.add(local_j * stripe.y_stride_elems),
                        eff_width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u8.add(local_j * stripe.u_stride_elems),
                        eff_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v8.add(local_j * stripe.v_stride_elems),
                        eff_width,
                    );
                    let da: &mut [MaybeUninit<u8>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a8.add(local_j * stripe.a_stride_elems),
                            eff_width,
                        )
                    } else {
                        &mut []
                    };
                    row_444_general_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        dy,
                        du,
                        dv,
                        da,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        eff_width,
                        fmt,
                        p,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// # Safety
    /// The caller must ensure `avx2` support and valid stripe bounds.
    #[target_feature(enable = "avx2")]
    unsafe fn run_444_avx2_stripe(
        stripe: &YuvStripe,
        cfg: &Avx2Config,
        fmt: RgbFmt,
        yuv_range: YuvRange,
    ) -> AvifResult<()> {
        // SAFETY: Target feature `avx2` is enabled on this function.
        let params = unsafe { build_params(cfg, fmt) };
        let p = &params;
        let has_a = if p.image_depth > 8 {
            !stripe.dst_a16.is_null()
        } else {
            !stripe.dst_a8.is_null()
        };
        let width = stripe.width;
        let ch = fmt.channels();
        let mut eff_width = width;
        let mut eff_rows = stripe.row_count;
        let tight_src = stripe.src_stride_elems == width * ch;
        let tight_dst = stripe.y_stride_elems == width
            && stripe.u_stride_elems == width
            && stripe.v_stride_elems == width
            && (!has_a || stripe.a_stride_elems == width);
        if eff_rows > 1 && tight_src && tight_dst {
            if let Some(total_px) = eff_width.checked_mul(eff_rows) {
                if total_px.checked_mul(ch).is_some() {
                    eff_width = total_px;
                    eff_rows = 1;
                }
            }
        }

        let fast_id_u8 = p.mode == Mode::Identity
            && p.rgb_depth == 8
            && p.image_depth == 8
            && yuv_range == YuvRange::Full;
        let fast_coeffs_u8 =
            matches!(p.mode, Mode::YuvCoefficients(..)) && p.rgb_depth == 8 && p.image_depth == 8;

        if fast_id_u8 {
            // SAFETY: `avx2` is enabled and stripe pointers cover `(0..eff_rows) x (0..eff_width)`.
            unsafe {
                return match (fmt, has_a) {
                    (RgbFmt::Rgba, true) => {
                        run_444_identity_u8_stripe::<0, true>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Rgba, false) => {
                        run_444_identity_u8_stripe::<0, false>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, true) => {
                        run_444_identity_u8_stripe::<1, true>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, false) => {
                        run_444_identity_u8_stripe::<1, false>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Rgb | RgbFmt::Bgr, _) => {
                        run_444_identity_u8_stripe::<2, false>(stripe, eff_width, eff_rows, fmt, p)
                    }
                };
            }
        }
        if fast_coeffs_u8 {
            // SAFETY: `avx2` is enabled and stripe pointers cover `(0..eff_rows) x (0..eff_width)`.
            unsafe {
                return match (fmt, has_a) {
                    (RgbFmt::Rgba, true) => {
                        run_444_u8_coeffs_stripe::<0, true>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Rgba, false) => {
                        run_444_u8_coeffs_stripe::<0, false>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, true) => {
                        run_444_u8_coeffs_stripe::<1, true>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, false) => {
                        run_444_u8_coeffs_stripe::<1, false>(stripe, eff_width, eff_rows, fmt, p)
                    }
                    (RgbFmt::Rgb | RgbFmt::Bgr, _) => {
                        run_444_u8_coeffs_stripe::<2, false>(stripe, eff_width, eff_rows, fmt, p)
                    }
                };
            }
        }
        // `(MODE_COEFFS, true, true)` returned early via `fast_coeffs_u8` above; prune that
        // unreachable arm so `run_444_general_stripe::<_, 0, true, true, _>` is not instantiated.
        macro_rules! with_444_fmt {
            ($mt:literal, $r8:literal, $i8:literal) => {
                match (fmt, has_a) {
                    (RgbFmt::Rgba, true) => run_444_general_stripe::<0, $mt, $r8, $i8, true>(
                        stripe, eff_width, eff_rows, fmt, p,
                    ),
                    (RgbFmt::Rgba, false) => run_444_general_stripe::<0, $mt, $r8, $i8, false>(
                        stripe, eff_width, eff_rows, fmt, p,
                    ),
                    (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, true) => {
                        run_444_general_stripe::<1, $mt, $r8, $i8, true>(
                            stripe, eff_width, eff_rows, fmt, p,
                        )
                    }
                    (RgbFmt::Bgra | RgbFmt::Argb | RgbFmt::Abgr, false) => {
                        run_444_general_stripe::<1, $mt, $r8, $i8, false>(
                            stripe, eff_width, eff_rows, fmt, p,
                        )
                    }
                    (RgbFmt::Rgb | RgbFmt::Bgr, _) => {
                        run_444_general_stripe::<2, $mt, $r8, $i8, false>(
                            stripe, eff_width, eff_rows, fmt, p,
                        )
                    }
                }
            };
        }
        macro_rules! with_444_depths {
            ($mt:literal) => {
                match (p.rgb_depth == 8, p.image_depth == 8) {
                    (true, true) => with_444_fmt!($mt, true, true),
                    (true, false) => with_444_fmt!($mt, true, false),
                    (false, true) => with_444_fmt!($mt, false, true),
                    (false, false) => with_444_fmt!($mt, false, false),
                }
            };
        }
        // SAFETY: `avx2` is enabled and stripe pointers cover `(0..eff_rows) x (0..eff_width)`.
        unsafe {
            match mode_tag(p.mode) {
                MODE_COEFFS => match (p.rgb_depth == 8, p.image_depth == 8) {
                    (true, true) => unreachable!(),
                    (true, false) => with_444_fmt!(0, true, false),
                    (false, true) => with_444_fmt!(0, false, true),
                    (false, false) => with_444_fmt!(0, false, false),
                },
                MODE_IDENTITY => with_444_depths!(1),
                MODE_YCGCO => with_444_depths!(2),
                _ => with_444_depths!(3),
            }
        }
    }

    /// # Safety
    /// Caller must ensure `avx2` support and valid stripe bounds.
    #[target_feature(enable = "avx2")]
    unsafe fn run_422_stripe<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        stripe: &YuvStripe,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let width = stripe.width;
        let uv_width = width.div_ceil(2);
        let src_req = width * fmt.channels();
        for local_j in 0..stripe.row_count {
            // SAFETY: Caller guarantees stripe bounds; `local_j < stripe.row_count`.
            unsafe {
                let (s8, s16): (&[u8], &[u16]) = if !RGB_8 {
                    (
                        &[],
                        std::slice::from_raw_parts(
                            stripe.src16.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                    )
                } else {
                    (
                        std::slice::from_raw_parts(
                            stripe.src8.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                        &[],
                    )
                };
                if !IMG_8 {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y16.add(local_j * stripe.y_stride_elems),
                        width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u16.add(local_j * stripe.u_stride_elems),
                        uv_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v16.add(local_j * stripe.v_stride_elems),
                        uv_width,
                    );
                    let da: &mut [MaybeUninit<u16>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a16.add(local_j * stripe.a_stride_elems),
                            width,
                        )
                    } else {
                        &mut []
                    };
                    row_422_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        dy,
                        du,
                        dv,
                        da,
                        width,
                        fmt,
                        p,
                    )?;
                } else {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y8.add(local_j * stripe.y_stride_elems),
                        width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u8.add(local_j * stripe.u_stride_elems),
                        uv_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v8.add(local_j * stripe.v_stride_elems),
                        uv_width,
                    );
                    let da: &mut [MaybeUninit<u8>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a8.add(local_j * stripe.a_stride_elems),
                            width,
                        )
                    } else {
                        &mut []
                    };
                    row_422_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        dy,
                        du,
                        dv,
                        da,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        width,
                        fmt,
                        p,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// # Safety
    /// The caller must ensure `avx2` support and valid stripe bounds.
    #[target_feature(enable = "avx2")]
    unsafe fn run_422_avx2_stripe(
        stripe: &YuvStripe,
        cfg: &Avx2Config,
        fmt: RgbFmt,
    ) -> AvifResult<()> {
        // SAFETY: Target feature `avx2` is enabled on this function.
        let params = unsafe { build_params(cfg, fmt) };
        let p = &params;
        let has_a = if p.image_depth > 8 {
            !stripe.dst_a16.is_null()
        } else {
            !stripe.dst_a8.is_null()
        };
        // SAFETY: `avx2` is enabled and stripe pointers cover `(0..stripe.row_count) x (0..width)`.
        unsafe {
            dispatch_spec!(
                fmt,
                mode_tag(p.mode),
                p.rgb_depth == 8,
                p.image_depth == 8,
                has_a,
                run_422_stripe(stripe, fmt, p)
            )
        }
    }

    /// # Safety
    /// Caller must ensure `avx2` support and valid stripe bounds (with `stripe.row_count` even
    /// unless this is the final stripe of an odd-height image).
    #[target_feature(enable = "avx2")]
    unsafe fn run_420_stripe<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        stripe: &YuvStripe,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let width = stripe.width;
        let uv_width = width.div_ceil(2);
        let src_req = width * fmt.channels();
        let paired_rows = stripe.row_count & !1;
        for local_j in (0..paired_rows).step_by(2) {
            let uv_j = local_j >> 1;
            // SAFETY: Caller guarantees stripe bounds; `local_j + 1 < stripe.row_count`.
            unsafe {
                let (s8_0, s8_1, s16_0, s16_1): (&[u8], &[u8], &[u16], &[u16]) = if !RGB_8 {
                    (
                        &[],
                        &[],
                        std::slice::from_raw_parts(
                            stripe.src16.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                        std::slice::from_raw_parts(
                            stripe.src16.add((local_j + 1) * stripe.src_stride_elems),
                            src_req,
                        ),
                    )
                } else {
                    (
                        std::slice::from_raw_parts(
                            stripe.src8.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                        std::slice::from_raw_parts(
                            stripe.src8.add((local_j + 1) * stripe.src_stride_elems),
                            src_req,
                        ),
                        &[],
                        &[],
                    )
                };
                if !IMG_8 {
                    let dy0 = std::slice::from_raw_parts_mut(
                        stripe.dst_y16.add(local_j * stripe.y_stride_elems),
                        width,
                    );
                    let dy1 = std::slice::from_raw_parts_mut(
                        stripe.dst_y16.add((local_j + 1) * stripe.y_stride_elems),
                        width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u16.add(uv_j * stripe.u_stride_elems),
                        uv_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v16.add(uv_j * stripe.v_stride_elems),
                        uv_width,
                    );
                    let (da0, da1): (&mut [MaybeUninit<u16>], &mut [MaybeUninit<u16>]) =
                        if KIND != 2 && HAS_A {
                            (
                                std::slice::from_raw_parts_mut(
                                    stripe.dst_a16.add(local_j * stripe.a_stride_elems),
                                    width,
                                ),
                                std::slice::from_raw_parts_mut(
                                    stripe.dst_a16.add((local_j + 1) * stripe.a_stride_elems),
                                    width,
                                ),
                            )
                        } else {
                            (&mut [], &mut [])
                        };
                    two_rows_420_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8_0,
                        s8_1,
                        s16_0,
                        s16_1,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        dy0,
                        dy1,
                        du,
                        dv,
                        da0,
                        da1,
                        width,
                        fmt,
                        p,
                    )?;
                } else {
                    let dy0 = std::slice::from_raw_parts_mut(
                        stripe.dst_y8.add(local_j * stripe.y_stride_elems),
                        width,
                    );
                    let dy1 = std::slice::from_raw_parts_mut(
                        stripe.dst_y8.add((local_j + 1) * stripe.y_stride_elems),
                        width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u8.add(uv_j * stripe.u_stride_elems),
                        uv_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v8.add(uv_j * stripe.v_stride_elems),
                        uv_width,
                    );
                    let (da0, da1): (&mut [MaybeUninit<u8>], &mut [MaybeUninit<u8>]) =
                        if KIND != 2 && HAS_A {
                            (
                                std::slice::from_raw_parts_mut(
                                    stripe.dst_a8.add(local_j * stripe.a_stride_elems),
                                    width,
                                ),
                                std::slice::from_raw_parts_mut(
                                    stripe.dst_a8.add((local_j + 1) * stripe.a_stride_elems),
                                    width,
                                ),
                            )
                        } else {
                            (&mut [], &mut [])
                        };
                    two_rows_420_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8_0,
                        s8_1,
                        s16_0,
                        s16_1,
                        dy0,
                        dy1,
                        du,
                        dv,
                        da0,
                        da1,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        width,
                        fmt,
                        p,
                    )?;
                }
            }
        }
        if stripe.row_count % 2 != 0 {
            let local_j = stripe.row_count - 1;
            let uv_j = local_j >> 1;
            // SAFETY: Caller guarantees stripe bounds; `local_j < stripe.row_count`.
            unsafe {
                let (s8, s16): (&[u8], &[u16]) = if !RGB_8 {
                    (
                        &[],
                        std::slice::from_raw_parts(
                            stripe.src16.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                    )
                } else {
                    (
                        std::slice::from_raw_parts(
                            stripe.src8.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                        &[],
                    )
                };
                if !IMG_8 {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y16.add(local_j * stripe.y_stride_elems),
                        width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u16.add(uv_j * stripe.u_stride_elems),
                        uv_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v16.add(uv_j * stripe.v_stride_elems),
                        uv_width,
                    );
                    let da: &mut [MaybeUninit<u16>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a16.add(local_j * stripe.a_stride_elems),
                            width,
                        )
                    } else {
                        &mut []
                    };
                    row_422_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        dy,
                        du,
                        dv,
                        da,
                        width,
                        fmt,
                        p,
                    )?;
                } else {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y8.add(local_j * stripe.y_stride_elems),
                        width,
                    );
                    let du = std::slice::from_raw_parts_mut(
                        stripe.dst_u8.add(uv_j * stripe.u_stride_elems),
                        uv_width,
                    );
                    let dv = std::slice::from_raw_parts_mut(
                        stripe.dst_v8.add(uv_j * stripe.v_stride_elems),
                        uv_width,
                    );
                    let da: &mut [MaybeUninit<u8>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a8.add(local_j * stripe.a_stride_elems),
                            width,
                        )
                    } else {
                        &mut []
                    };
                    row_422_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        dy,
                        du,
                        dv,
                        da,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        width,
                        fmt,
                        p,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// # Safety
    /// The caller must ensure `avx2` support and valid stripe bounds.
    #[target_feature(enable = "avx2")]
    unsafe fn run_420_avx2_stripe(
        stripe: &YuvStripe,
        cfg: &Avx2Config,
        fmt: RgbFmt,
    ) -> AvifResult<()> {
        // SAFETY: Target feature `avx2` is enabled on this function.
        let params = unsafe { build_params(cfg, fmt) };
        let p = &params;
        let has_a = if p.image_depth > 8 {
            !stripe.dst_a16.is_null()
        } else {
            !stripe.dst_a8.is_null()
        };
        // SAFETY: `avx2` is enabled and stripe pointers cover `(0..stripe.row_count) x (0..width)`.
        unsafe {
            dispatch_spec!(
                fmt,
                mode_tag(p.mode),
                p.rgb_depth == 8,
                p.image_depth == 8,
                has_a,
                run_420_stripe(stripe, fmt, p)
            )
        }
    }

    /// # Safety
    /// Caller must ensure `avx2` support and that `stripe` pointers cover
    /// `(0..eff_rows) x (0..eff_width)` for read (source) and exclusive write (destination).
    #[target_feature(enable = "avx2")]
    unsafe fn run_400_stripe<
        const KIND: u8,
        const MODE_TAG: u8,
        const RGB_8: bool,
        const IMG_8: bool,
        const HAS_A: bool,
    >(
        stripe: &YuvStripe,
        eff_width: usize,
        eff_rows: usize,
        fmt: RgbFmt,
        p: &Avx2Params,
    ) -> AvifResult<()> {
        let ch = fmt.channels();
        let src_req = eff_width.checked_mul(ch).ok_or(AvifError::NoContent)?;
        for local_j in 0..eff_rows {
            // SAFETY: Caller guarantees stripe bounds; `local_j < eff_rows`.
            unsafe {
                let (s8, s16): (&[u8], &[u16]) = if !RGB_8 {
                    (
                        &[],
                        std::slice::from_raw_parts(
                            stripe.src16.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                    )
                } else {
                    (
                        std::slice::from_raw_parts(
                            stripe.src8.add(local_j * stripe.src_stride_elems),
                            src_req,
                        ),
                        &[],
                    )
                };
                if !IMG_8 {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y16.add(local_j * stripe.y_stride_elems),
                        eff_width,
                    );
                    let da: &mut [MaybeUninit<u16>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a16.add(local_j * stripe.a_stride_elems),
                            eff_width,
                        )
                    } else {
                        &mut []
                    };
                    row_400_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        &mut [],
                        &mut [],
                        dy,
                        da,
                        eff_width,
                        fmt,
                        p,
                    )?;
                } else {
                    let dy = std::slice::from_raw_parts_mut(
                        stripe.dst_y8.add(local_j * stripe.y_stride_elems),
                        eff_width,
                    );
                    let da: &mut [MaybeUninit<u8>] = if KIND != 2 && HAS_A {
                        std::slice::from_raw_parts_mut(
                            stripe.dst_a8.add(local_j * stripe.a_stride_elems),
                            eff_width,
                        )
                    } else {
                        &mut []
                    };
                    row_400_spec::<KIND, MODE_TAG, RGB_8, IMG_8, HAS_A>(
                        s8,
                        s16,
                        dy,
                        da,
                        &mut [],
                        &mut [],
                        eff_width,
                        fmt,
                        p,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// # Safety
    /// The caller must ensure `avx2` support and valid stripe bounds.
    #[target_feature(enable = "avx2")]
    unsafe fn run_400_avx2_stripe(
        stripe: &YuvStripe,
        cfg: &Avx2Config,
        fmt: RgbFmt,
    ) -> AvifResult<()> {
        // SAFETY: Target feature `avx2` is enabled on this function.
        let params = unsafe { build_params(cfg, fmt) };
        let p = &params;
        let has_a = if p.image_depth > 8 {
            !stripe.dst_a16.is_null()
        } else {
            !stripe.dst_a8.is_null()
        };
        let width = stripe.width;
        let ch = fmt.channels();
        let mut eff_width = width;
        let mut eff_rows = stripe.row_count;
        let tight_src = stripe.src_stride_elems == width * ch;
        let tight_dst =
            stripe.y_stride_elems == width && (!has_a || stripe.a_stride_elems == width);
        if eff_rows > 1 && tight_src && tight_dst {
            if let Some(total_px) = eff_width.checked_mul(eff_rows) {
                if total_px.checked_mul(ch).is_some() {
                    eff_width = total_px;
                    eff_rows = 1;
                }
            }
        }
        // SAFETY: `avx2` is enabled and stripe pointers cover `(0..eff_rows) x (0..eff_width)`.
        unsafe {
            dispatch_spec!(
                fmt,
                mode_tag(p.mode),
                p.rgb_depth == 8,
                p.image_depth == 8,
                has_a,
                run_400_stripe(stripe, eff_width, eff_rows, fmt, p)
            )
        }
    }

    struct RawPlanePtrs {
        ptr8: *mut MaybeUninit<u8>,
        ptr16: *mut MaybeUninit<u16>,
        stride_elems: usize,
    }

    impl RawPlanePtrs {
        fn null() -> Self {
            Self {
                ptr8: std::ptr::null_mut(),
                ptr16: std::ptr::null_mut(),
                stride_elems: 0,
            }
        }
    }

    enum UninitHolder {
        U8(UninitPlane8),
        U16(UninitPlane16),
    }

    impl UninitHolder {
        fn alloc(image: &image::Image, plane: Plane) -> AvifResult<(Self, RawPlanePtrs)> {
            if image.depth == 8 {
                let mut up = UninitPlane8::for_image(image, plane)?;
                let stride_elems = up.stride_elems();
                let ptr8 = up.spare_all_mut()?.as_mut_ptr();
                Ok((
                    Self::U8(up),
                    RawPlanePtrs {
                        ptr8,
                        ptr16: std::ptr::null_mut(),
                        stride_elems,
                    },
                ))
            } else {
                let mut up = UninitPlane16::for_image(image, plane)?;
                let stride_elems = up.stride_elems();
                let ptr16 = up.spare_all_mut()?.as_mut_ptr();
                Ok((
                    Self::U16(up),
                    RawPlanePtrs {
                        ptr8: std::ptr::null_mut(),
                        ptr16,
                        stride_elems,
                    },
                ))
            }
        }

        /// # Safety
        /// Caller must have initialized all logical samples in this plane.
        unsafe fn finish_into(
            self,
            image: &mut image::Image,
            plane: Plane,
            pad_value: u16,
        ) -> AvifResult<()> {
            match self {
                Self::U8(up) => {
                    // SAFETY: Forwarded from caller.
                    unsafe { up.finish_into(image, plane, pad_value) }
                }
                Self::U16(up) => {
                    // SAFETY: Forwarded from caller.
                    unsafe { up.finish_into(image, plane, pad_value) }
                }
            }
        }
    }

    /// # Safety
    /// Caller must ensure runtime `avx2` support, `rgb.width > 0 && rgb.height > 0`, and that
    /// source pointer bounds for `rgb` and destination plane pointer bounds are valid for
    /// `(0..height) x (0..width)` (and corresponding chroma dimensions for 4:2:2/4:2:0).
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_yuv_stripes(
        rgb: &rgb::Image,
        cfg: &Avx2Config,
        fmt: RgbFmt,
        yuv_format: PixelFormat,
        yuv_range: YuvRange,
        dst_y: &RawPlanePtrs,
        dst_u: &RawPlanePtrs,
        dst_v: &RawPlanePtrs,
        dst_a: Option<&RawPlanePtrs>,
        threads: u32,
    ) -> AvifResult<()> {
        let width = usize_from_u32(rgb.width)?;
        let height = usize_from_u32(rgb.height)?;
        let ch = fmt.channels();
        let src_row_bytes = usize_from_u32(rgb.row_bytes)?;
        let (src8_base, src16_base, src_stride_elems) = if cfg.rgb_depth > 8 {
            if src_row_bytes % 2 != 0 {
                return AvifError::no_content();
            }
            let stride_u16 = src_row_bytes / 2;
            let req = (height - 1)
                .checked_mul(stride_u16)
                .and_then(|x| x.checked_add(width * ch))
                .ok_or(AvifError::NoContent)?;
            let s16 = rgb
                .pixels
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .slice16(0, u32_from_usize(req)?)?;
            (std::ptr::null(), s16.as_ptr(), stride_u16)
        } else {
            let req = (height - 1)
                .checked_mul(src_row_bytes)
                .and_then(|x| x.checked_add(width * ch))
                .ok_or(AvifError::NoContent)?;
            let s8 = rgb
                .pixels
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .slice(0, u32_from_usize(req)?)?;
            (s8.as_ptr(), std::ptr::null(), src_row_bytes)
        };

        let do_mt = threads > 1 && width.saturating_mul(height) >= rgb::MIN_MT_PIXELS_RGB_TO_YUV;
        let max_workers = (threads as usize).min(rgb::MAX_RGB_TO_YUV_THREADS);
        let row_worker_cap = height / rgb::MIN_MT_ROWS_PER_WORKER;
        let workers =
            if do_mt && row_worker_cap >= 2 { max_workers.min(row_worker_cap) } else { 1 };

        let (da_ptr8, da_ptr16, a_stride) = match dst_a {
            Some(pa) => (pa.ptr8, pa.ptr16, pa.stride_elems),
            None => (std::ptr::null_mut(), std::ptr::null_mut(), 0),
        };

        let make_stripe = |row_start: usize, uv_start: usize, row_count: usize| -> YuvStripe {
            let src8 = if src8_base.is_null() {
                std::ptr::null()
            } else {
                // SAFETY: `row_start <= height` within validated `s8`.
                unsafe { src8_base.add(row_start * src_stride_elems) }
            };
            let src16 = if src16_base.is_null() {
                std::ptr::null()
            } else {
                // SAFETY: `row_start <= height` within validated `s16`.
                unsafe { src16_base.add(row_start * src_stride_elems) }
            };
            let dst_y8 = if dst_y.ptr8.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `row_start <= height` within `dst_y` allocation.
                unsafe { dst_y.ptr8.add(row_start * dst_y.stride_elems) }
            };
            let dst_u8 = if dst_u.ptr8.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `uv_start <= uv_height` within `dst_u` allocation.
                unsafe { dst_u.ptr8.add(uv_start * dst_u.stride_elems) }
            };
            let dst_v8 = if dst_v.ptr8.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `uv_start <= uv_height` within `dst_v` allocation.
                unsafe { dst_v.ptr8.add(uv_start * dst_v.stride_elems) }
            };
            let dst_a8 = if da_ptr8.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `row_start <= height` within `dst_a` allocation.
                unsafe { da_ptr8.add(row_start * a_stride) }
            };
            let dst_y16 = if dst_y.ptr16.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `row_start <= height` within `dst_y` allocation.
                unsafe { dst_y.ptr16.add(row_start * dst_y.stride_elems) }
            };
            let dst_u16 = if dst_u.ptr16.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `uv_start <= uv_height` within `dst_u` allocation.
                unsafe { dst_u.ptr16.add(uv_start * dst_u.stride_elems) }
            };
            let dst_v16 = if dst_v.ptr16.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `uv_start <= uv_height` within `dst_v` allocation.
                unsafe { dst_v.ptr16.add(uv_start * dst_v.stride_elems) }
            };
            let dst_a16 = if da_ptr16.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `row_start <= height` within `dst_a` allocation.
                unsafe { da_ptr16.add(row_start * a_stride) }
            };
            YuvStripe {
                src8,
                src16,
                src_stride_elems,
                dst_y8,
                dst_u8,
                dst_v8,
                dst_a8,
                dst_y16,
                dst_u16,
                dst_v16,
                dst_a16,
                y_stride_elems: dst_y.stride_elems,
                u_stride_elems: dst_u.stride_elems,
                v_stride_elems: dst_v.stride_elems,
                a_stride_elems: a_stride,
                width,
                row_count,
            }
        };

        let exec_stripe = |stripe: &YuvStripe| -> AvifResult<()> {
            // SAFETY: Caller ensures `avx2` support and valid stripe pointers.
            unsafe {
                match yuv_format {
                    PixelFormat::Yuv444 => run_444_avx2_stripe(stripe, cfg, fmt, yuv_range),
                    PixelFormat::Yuv422 => run_422_avx2_stripe(stripe, cfg, fmt),
                    PixelFormat::Yuv420 => run_420_avx2_stripe(stripe, cfg, fmt),
                    PixelFormat::Yuv400 => run_400_avx2_stripe(stripe, cfg, fmt),
                    _ => AvifError::reformat_failed(),
                }
            }
        };

        if workers <= 1 {
            return exec_stripe(&make_stripe(0, 0, height));
        }

        let mut stripes = [make_stripe(0, 0, 0); rgb::MAX_RGB_TO_YUV_THREADS];
        if yuv_format == PixelFormat::Yuv420 {
            let pairs = height / 2;
            let pairs_per = pairs / workers;
            let extra = pairs % workers;
            let mut pair_start = 0usize;
            for (idx, dst_stripe) in stripes[..workers].iter_mut().enumerate() {
                let stripe_pairs = pairs_per + usize::from(idx < extra);
                let is_last = idx + 1 == workers;
                let row_count = stripe_pairs * 2 + if is_last { height % 2 } else { 0 };
                let row_start = pair_start * 2;
                *dst_stripe = make_stripe(row_start, pair_start, row_count);
                pair_start += stripe_pairs;
            }
        } else {
            let rows_per = height / workers;
            let extra = height % workers;
            let mut row_start = 0usize;
            for (idx, dst_stripe) in stripes[..workers].iter_mut().enumerate() {
                let row_count = rows_per + usize::from(idx < extra);
                *dst_stripe = make_stripe(row_start, row_start, row_count);
                row_start += row_count;
            }
        }

        let (first, rest) = stripes[..workers]
            .split_first()
            .ok_or(AvifError::ReformatFailed)?;
        let mut first_res = Ok(());
        let mut rest_err: AvifResult<()> = Ok(());
        std::thread::scope(|scope| -> AvifResult<()> {
            let mut handles: [Option<std::thread::ScopedJoinHandle<'_, AvifResult<()>>>; 7] =
                std::array::from_fn(|_| None);
            for (slot, &stripe) in handles[..rest.len()].iter_mut().zip(rest.iter()) {
                let handle = std::thread::Builder::new()
                    .spawn_scoped(scope, move || {
                        let stripe = stripe;
                        exec_stripe(&stripe)
                    })
                    .map_err(|_| AvifError::ReformatFailed)?;
                *slot = Some(handle);
            }
            first_res = exec_stripe(first);
            for slot in &mut handles[..rest.len()] {
                if let Some(handle) = slot.take() {
                    let res = handle.join().unwrap_or(AvifError::reformat_failed());
                    if rest_err.is_ok() && res.is_err() {
                        rest_err = res;
                    }
                }
            }
            Ok(())
        })?;
        first_res?;
        rest_err
    }

    fn avx2_dispatch_precheck(rgb: &rgb::Image, image: &image::Image) -> Option<RgbFmt> {
        if rgb.width != image.width
            || rgb.height != image.height
            || !matches!(rgb.depth, 8 | 10 | 12 | 16)
            || !matches!(image.depth, 8 | 10 | 12 | 16)
            || !rgb.has_valid_pixels()
            || !matches!(
                image.yuv_format,
                PixelFormat::Yuv444
                    | PixelFormat::Yuv422
                    | PixelFormat::Yuv420
                    | PixelFormat::Yuv400
            )
        {
            return None;
        }
        let fmt = RgbFmt::from_format(rgb.format)?;
        if !std::arch::is_x86_feature_detected!("avx2") {
            return None;
        }
        Some(fmt)
    }

    pub(super) fn try_rgb_to_yuv_uninit_with_threads(
        rgb: &rgb::Image,
        image: &mut image::Image,
        alloc_alpha: bool,
        threads: u32,
    ) -> AvifResult<Option<bool>> {
        let Some(fmt) = avx2_dispatch_precheck(rgb, image) else {
            return Ok(None);
        };
        let cfg = build_config(rgb, image);
        let fuse_alpha = alloc_alpha && can_fuse_alpha_base(rgb, image);
        let yuv_range = image.yuv_range;
        let yuv_format = image.yuv_format;
        let max_alpha = image.max_channel();

        let (hold_y, py) = UninitHolder::alloc(image, Plane::Y)?;
        let a_alloc = if fuse_alpha { Some(UninitHolder::alloc(image, Plane::A)?) } else { None };
        let (hold_a, pa) = match a_alloc {
            Some((h, ptrs)) => (Some(h), Some(ptrs)),
            None => (None, None),
        };

        if yuv_format == PixelFormat::Yuv400 {
            let null = RawPlanePtrs::null();
            // SAFETY: `UninitHolder::alloc` reserved full `(0..height) x (0..width)` spare
            // buffers, `avx2` support was verified above, and all threads join before return.
            unsafe {
                run_yuv_stripes(
                    rgb,
                    &cfg,
                    fmt,
                    yuv_format,
                    yuv_range,
                    &py,
                    &null,
                    &null,
                    pa.as_ref(),
                    threads,
                )?;
            }
            // SAFETY: `run_yuv_stripes` initialized all `(0..height) x (0..width)` Y samples.
            unsafe { hold_y.finish_into(image, Plane::Y, 0)? };
            if let Some(ha) = hold_a {
                // SAFETY: `run_yuv_stripes` initialized all `(0..height) x (0..width)` A samples.
                if let Err(err) = unsafe { ha.finish_into(image, Plane::A, max_alpha) } {
                    image.free_planes(&[Plane::Y]);
                    return Err(err);
                }
            }
            image.free_planes(&[Plane::U, Plane::V]);
            return Ok(Some(fuse_alpha));
        }

        let (hold_u, pu) = UninitHolder::alloc(image, Plane::U)?;
        let (hold_v, pv) = UninitHolder::alloc(image, Plane::V)?;
        // SAFETY: `UninitHolder::alloc` reserved full plane spare buffers for Y, U, V (and A),
        // `avx2` support was verified above, and all scoped threads join before return.
        unsafe {
            run_yuv_stripes(
                rgb,
                &cfg,
                fmt,
                yuv_format,
                yuv_range,
                &py,
                &pu,
                &pv,
                pa.as_ref(),
                threads,
            )?;
        }
        // SAFETY: `run_yuv_stripes` initialized all logical Y samples.
        unsafe { hold_y.finish_into(image, Plane::Y, 0)? };
        // SAFETY: `run_yuv_stripes` initialized all logical U samples.
        if let Err(err) = unsafe { hold_u.finish_into(image, Plane::U, 0) } {
            image.free_planes(&[Plane::Y]);
            return Err(err);
        }
        // SAFETY: `run_yuv_stripes` initialized all logical V samples.
        if let Err(err) = unsafe { hold_v.finish_into(image, Plane::V, 0) } {
            image.free_planes(&[Plane::Y, Plane::U]);
            return Err(err);
        }
        if let Some(ha) = hold_a {
            // SAFETY: `run_yuv_stripes` initialized all logical A samples.
            if let Err(err) = unsafe { ha.finish_into(image, Plane::A, max_alpha) } {
                image.free_planes(&[Plane::Y, Plane::U, Plane::V]);
                return Err(err);
            }
        }
        Ok(Some(fuse_alpha))
    }

    #[derive(Clone, Copy)]
    struct GrayConfig {
        rgb_max_f: f32,
        bias_y: f32,
        range_y: f32,
        yuv_max: u16,
    }

    struct GrayParams {
        rgb_max_f: f32,
        bias_y: f32,
        range_y: f32,
        yuv_max: u16,
        v_rgb_max: __m256,
        v_bias_y: __m256,
        v_range_y: __m256,
        v_half: __m256,
        v_max_u16: __m256i,
        mask_ffff: __m256i,
        mask_lo16: __m128i,
    }

    /// # Safety
    /// Caller must ensure `avx2` support and that source/destination buffers cover `width` items.
    #[target_feature(enable = "avx2")]
    unsafe fn row_gray_avx2_spec<
        const CH: usize,
        const OFF: usize,
        const RGB_8: bool,
        const IMG_8: bool,
    >(
        s8p: *const u8,
        s16p: *const u16,
        y8p: *mut u8,
        y16p: *mut u16,
        width: usize,
        gp: &GrayParams,
    ) {
        let mut i = 0usize;
        while i + 16 <= width {
            // SAFETY: Loop guard `i + 16 <= width` and caller buffer checks guarantee readable
            // input samples and writable output pixels for `[i..i + 16)`.
            unsafe {
                let (g0, g1) = if RGB_8 {
                    if CH == 1 {
                        (
                            _mm256_cvtepu8_epi32(_mm_loadl_epi64(s8p.add(i).cast())),
                            _mm256_cvtepu8_epi32(_mm_loadl_epi64(s8p.add(i + 8).cast())),
                        )
                    } else {
                        let r0 = _mm_loadu_si128(s8p.add(i * 2).cast());
                        let r1 = _mm_loadu_si128(s8p.add((i + 8) * 2).cast());
                        if OFF == 0 {
                            (
                                _mm256_cvtepu16_epi32(_mm_and_si128(r0, gp.mask_lo16)),
                                _mm256_cvtepu16_epi32(_mm_and_si128(r1, gp.mask_lo16)),
                            )
                        } else {
                            (
                                _mm256_cvtepu16_epi32(_mm_srli_epi16(r0, 8)),
                                _mm256_cvtepu16_epi32(_mm_srli_epi16(r1, 8)),
                            )
                        }
                    }
                } else if CH == 1 {
                    (
                        _mm256_cvtepu16_epi32(_mm_loadu_si128(s16p.add(i).cast())),
                        _mm256_cvtepu16_epi32(_mm_loadu_si128(s16p.add(i + 8).cast())),
                    )
                } else {
                    let r0 = _mm256_loadu_si256(s16p.add(i * 2).cast());
                    let r1 = _mm256_loadu_si256(s16p.add((i + 8) * 2).cast());
                    if OFF == 0 {
                        (
                            _mm256_and_si256(r0, gp.mask_ffff),
                            _mm256_and_si256(r1, gp.mask_ffff),
                        )
                    } else {
                        (_mm256_srli_epi32(r0, 16), _mm256_srli_epi32(r1, 16))
                    }
                };
                let f0 = _mm256_div_ps(_mm256_cvtepi32_ps(g0), gp.v_rgb_max);
                let f1 = _mm256_div_ps(_mm256_cvtepi32_ps(g1), gp.v_rgb_max);
                let q0 = quant_i32(f0, gp.v_range_y, gp.v_bias_y, gp.v_half);
                let q1 = quant_i32(f1, gp.v_range_y, gp.v_bias_y, gp.v_half);
                if !IMG_8 {
                    store16_u16_ptr(y16p.add(i), q0, q1, gp.v_max_u16);
                } else {
                    store16_u8_ptr(y8p.add(i), q0, q1);
                }
            }
            i += 16;
        }
        if i + 8 <= width {
            // SAFETY: Loop guard `i + 8 <= width` and caller buffer checks guarantee readable
            // input samples and writable output pixels for `[i..i + 8)`.
            unsafe {
                let g0 = if RGB_8 {
                    if CH == 1 {
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(s8p.add(i).cast()))
                    } else {
                        let r0 = _mm_loadu_si128(s8p.add(i * 2).cast());
                        if OFF == 0 {
                            _mm256_cvtepu16_epi32(_mm_and_si128(r0, gp.mask_lo16))
                        } else {
                            _mm256_cvtepu16_epi32(_mm_srli_epi16(r0, 8))
                        }
                    }
                } else if CH == 1 {
                    _mm256_cvtepu16_epi32(_mm_loadu_si128(s16p.add(i).cast()))
                } else {
                    let r0 = _mm256_loadu_si256(s16p.add(i * 2).cast());
                    if OFF == 0 {
                        _mm256_and_si256(r0, gp.mask_ffff)
                    } else {
                        _mm256_srli_epi32(r0, 16)
                    }
                };
                let f0 = _mm256_div_ps(_mm256_cvtepi32_ps(g0), gp.v_rgb_max);
                let q0 = quant_i32(f0, gp.v_range_y, gp.v_bias_y, gp.v_half);
                if !IMG_8 {
                    store8_u16_ptr(y16p.add(i), q0, gp.v_max_u16);
                } else {
                    store8_u8_ptr(y8p.add(i), q0);
                }
            }
            i += 8;
        }
        for x in i..width {
            // SAFETY: `x < width` and caller buffer checks guarantee valid reads and writes.
            unsafe {
                let gray_pixel = if RGB_8 {
                    *s8p.add(x * CH + OFF) as f32 / gp.rgb_max_f
                } else {
                    *s16p.add(x * CH + OFF) as f32 / gp.rgb_max_f
                };
                let y_val = to_unorm(gp.bias_y, gp.range_y, gp.yuv_max, gray_pixel);
                if !IMG_8 {
                    *y16p.add(x) = y_val;
                } else {
                    *y8p.add(x) = y_val as u8;
                }
            }
        }
    }

    /// # Safety
    /// Caller must ensure `avx2` support and that source/destination buffers cover
    /// `(0..eff_rows) x (0..eff_width)` with the given strides.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_gray_avx2<
        const CH: usize,
        const OFF: usize,
        const RGB_8: bool,
        const IMG_8: bool,
    >(
        src8_base: *const u8,
        src16_base: *const u16,
        src_stride_elems: usize,
        dst_y: &RawPlanePtrs,
        eff_width: usize,
        eff_rows: usize,
        gp: &GrayParams,
    ) {
        let y8_base = dst_y.ptr8.cast::<u8>();
        let y16_base = dst_y.ptr16.cast::<u16>();
        for row in 0..eff_rows {
            // SAFETY: `row < eff_rows` and caller validated all row/column bounds.
            unsafe {
                let s8p =
                    if RGB_8 { src8_base.add(row * src_stride_elems) } else { std::ptr::null() };
                let s16p =
                    if !RGB_8 { src16_base.add(row * src_stride_elems) } else { std::ptr::null() };
                let y8p = if IMG_8 {
                    y8_base.add(row * dst_y.stride_elems)
                } else {
                    std::ptr::null_mut()
                };
                let y16p = if !IMG_8 {
                    y16_base.add(row * dst_y.stride_elems)
                } else {
                    std::ptr::null_mut()
                };
                row_gray_avx2_spec::<CH, OFF, RGB_8, IMG_8>(s8p, s16p, y8p, y16p, eff_width, gp);
            }
        }
    }

    /// # Safety
    /// Caller must ensure `avx2` support and that source/destination buffers cover
    /// `(0..eff_rows) x (0..eff_width)` with the given strides.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_gray_avx2_entry(
        ch: usize,
        gray_off: usize,
        rgb_8: bool,
        img_8: bool,
        src8_base: *const u8,
        src16_base: *const u16,
        src_stride_elems: usize,
        dst_y: &RawPlanePtrs,
        eff_width: usize,
        eff_rows: usize,
        cfg: GrayConfig,
    ) -> Option<()> {
        let gp = GrayParams {
            rgb_max_f: cfg.rgb_max_f,
            bias_y: cfg.bias_y,
            range_y: cfg.range_y,
            yuv_max: cfg.yuv_max,
            v_rgb_max: _mm256_set1_ps(cfg.rgb_max_f),
            v_bias_y: _mm256_set1_ps(cfg.bias_y),
            v_range_y: _mm256_set1_ps(cfg.range_y),
            v_half: _mm256_set1_ps(0.5),
            v_max_u16: _mm256_set1_epi16(cfg.yuv_max as i16),
            mask_ffff: _mm256_set1_epi32(0xFFFF),
            mask_lo16: _mm_set1_epi16(0x00FF),
        };
        // SAFETY: Caller guarantees `avx2` support and validated source/destination spans.
        unsafe {
            match (ch, gray_off, rgb_8, img_8) {
                (1, 0, true, true) => {
                    run_gray_avx2::<1, 0, true, true>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (1, 0, true, false) => {
                    run_gray_avx2::<1, 0, true, false>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (1, 0, false, true) => {
                    run_gray_avx2::<1, 0, false, true>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (1, 0, false, false) => {
                    run_gray_avx2::<1, 0, false, false>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 0, true, true) => {
                    run_gray_avx2::<2, 0, true, true>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 0, true, false) => {
                    run_gray_avx2::<2, 0, true, false>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 0, false, true) => {
                    run_gray_avx2::<2, 0, false, true>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 0, false, false) => {
                    run_gray_avx2::<2, 0, false, false>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 1, true, true) => {
                    run_gray_avx2::<2, 1, true, true>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 1, true, false) => {
                    run_gray_avx2::<2, 1, true, false>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 1, false, true) => {
                    run_gray_avx2::<2, 1, false, true>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                (2, 1, false, false) => {
                    run_gray_avx2::<2, 1, false, false>(
                        src8_base,
                        src16_base,
                        src_stride_elems,
                        dst_y,
                        eff_width,
                        eff_rows,
                        &gp,
                    );
                }
                _ => return None,
            }
        }
        Some(())
    }

    pub(super) fn try_rgb_gray_to_yuv(
        rgb: &rgb::Image,
        image: &mut image::Image,
    ) -> AvifResult<Option<()>> {
        let (ch, gray_off) = match rgb.format {
            Format::Gray => (1usize, 0usize),
            Format::GrayA => (2usize, 0usize),
            Format::AGray => (2usize, 1usize),
            _ => return Ok(None),
        };
        if rgb.width != image.width
            || rgb.height != image.height
            || image.width == 0
            || image.height == 0
            || !matches!(rgb.depth, 8 | 10 | 12 | 16)
            || !matches!(image.depth, 8 | 10 | 12 | 16)
            || !rgb.has_valid_pixels()
        {
            return Ok(None);
        }
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Ok(None);
        }
        let width = usize_from_u32(image.width)?;
        let height = usize_from_u32(image.height)?;
        let src_row_bytes = usize_from_u32(rgb.row_bytes)?;
        let (src8_base, src16_base, src_stride_elems) = if rgb.depth > 8 {
            if src_row_bytes % 2 != 0 {
                return AvifError::no_content();
            }
            let stride_u16 = src_row_bytes / 2;
            let req = (height - 1)
                .checked_mul(stride_u16)
                .and_then(|x| x.checked_add(width * ch))
                .ok_or(AvifError::NoContent)?;
            let s16 = rgb
                .pixels
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .slice16(0, u32_from_usize(req)?)?;
            (std::ptr::null(), s16.as_ptr(), stride_u16)
        } else {
            let req = (height - 1)
                .checked_mul(src_row_bytes)
                .and_then(|x| x.checked_add(width * ch))
                .ok_or(AvifError::NoContent)?;
            let s8 = rgb
                .pixels
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .slice(0, u32_from_usize(req)?)?;
            (s8.as_ptr(), std::ptr::null(), src_row_bytes)
        };
        let image_depth = image.depth;
        let (hold_y, py) = UninitHolder::alloc(image, Plane::Y)?;

        let mut eff_width = width;
        let mut eff_rows = height;
        if eff_rows > 1 && src_stride_elems == width * ch && py.stride_elems == width {
            if let Some(total_px) = eff_width.checked_mul(eff_rows) {
                if total_px.checked_mul(ch).is_some() {
                    eff_width = total_px;
                    eff_rows = 1;
                }
            }
        }

        let (bias_y, range_y) = bias_and_range_y(image);
        let cfg = GrayConfig {
            rgb_max_f: rgb.max_channel_f(),
            bias_y,
            range_y,
            yuv_max: image.max_channel(),
        };
        // SAFETY: CPU support for `avx2` was verified above and all source/destination
        // buffer spans were bounds-validated before constructing pointers.
        let handled = unsafe {
            run_gray_avx2_entry(
                ch,
                gray_off,
                rgb.depth == 8,
                image_depth == 8,
                src8_base,
                src16_base,
                src_stride_elems,
                &py,
                eff_width,
                eff_rows,
                cfg,
            )
        };
        if handled.is_some() {
            // SAFETY: `run_gray_avx2_entry` initialized all `(0..height) x (0..width)` Y samples.
            unsafe { hold_y.finish_into(image, Plane::Y, 0)? };
        }
        Ok(handled)
    }
}

#[inline(always)]
fn to_unorm(bias_y: f32, range_y: f32, max_channel: u16, v: f32) -> u16 {
    clamp_i32((0.5 + (v * range_y + bias_y)) as i32, 0, max_channel as i32) as u16
}

fn validate_alpha_import(rgb: &rgb::Image, image: &image::Image) -> AvifResult<()> {
    if !rgb.has_alpha()
        || rgb.width != image.width
        || rgb.height != image.height
        || rgb.format == Format::Rgba1010102
        || !rgb.has_valid_pixels()
    {
        return AvifError::invalid_argument();
    }
    Ok(())
}

pub(crate) fn import_alpha_uninit(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<()> {
    validate_alpha_import(rgb, image)?;
    let width = usize_from_u32(image.width)?;
    let height = image.height;
    let height_usize = usize_from_u32(height)?;
    let src_alpha_offset = rgb.format.alpha_offset();
    let channel_count = rgb.format.channel_count() as usize;
    let max_channel = image.max_channel();
    let src_max_channel_f = rgb.max_channel_f();
    if image.depth == 8 {
        let mut dst = UninitPlane8::for_image(image, Plane::A)?;
        if rgb.depth == 8 {
            let fast_4ch = matches!(
                rgb.format,
                Format::Rgba | Format::Bgra | Format::Argb | Format::Abgr
            );
            if fast_4ch {
                let src_row_bytes = usize_from_u32(rgb.row_bytes)?;
                let src_req = checked_mul!(width, 4)?;
                let total_u8 =
                    checked_add!(checked_mul!(height_usize - 1, src_row_bytes)?, src_req)?;
                let src_all = rgb
                    .pixels
                    .as_ref()
                    .ok_or(AvifError::NoContent)?
                    .slice(0, u32_from_usize(total_u8)?)?;
                let dst_stride = dst.stride_elems();
                let dst_spare = dst.spare_all_mut()?;
                extract_alpha_4ch_stripe_u8(
                    src_all,
                    src_row_bytes,
                    0,
                    height_usize,
                    dst_spare,
                    dst_stride,
                    width,
                    src_alpha_offset,
                );
            } else {
                for y in 0..height {
                    let src_row = rgb.row(y)?;
                    let dst_row = dst.spare_row_mut(y)?;
                    for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                        cell.write(src_row[(x * channel_count) + src_alpha_offset]);
                    }
                }
            }
        } else {
            for y in 0..height {
                let src_row = rgb.row16(y)?;
                let dst_row = dst.spare_row_mut(y)?;
                for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                    let v = src_row[(x * channel_count) + src_alpha_offset];
                    cell.write(
                        rgb::Image::rescale_alpha_value(v, src_max_channel_f, max_channel) as u8,
                    );
                }
            }
        }
        // SAFETY: Every `(y, x)` in `(0..height) x (0..width)` was initialized above.
        unsafe { dst.finish_into(image, Plane::A, max_channel) }
    } else {
        let mut dst = UninitPlane16::for_image(image, Plane::A)?;
        if rgb.depth == image.depth {
            for y in 0..height {
                let src_row = rgb.row16(y)?;
                let dst_row = dst.spare_row_mut(y)?;
                for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                    cell.write(src_row[(x * channel_count) + src_alpha_offset]);
                }
            }
        } else if rgb.depth > 8 {
            for y in 0..height {
                let src_row = rgb.row16(y)?;
                let dst_row = dst.spare_row_mut(y)?;
                for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                    let v = src_row[(x * channel_count) + src_alpha_offset];
                    cell.write(rgb::Image::rescale_alpha_value(
                        v,
                        src_max_channel_f,
                        max_channel,
                    ));
                }
            }
        } else {
            for y in 0..height {
                let src_row = rgb.row(y)?;
                let dst_row = dst.spare_row_mut(y)?;
                for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                    let v = src_row[(x * channel_count) + src_alpha_offset] as u16;
                    cell.write(rgb::Image::rescale_alpha_value(
                        v,
                        src_max_channel_f,
                        max_channel,
                    ));
                }
            }
        }
        // SAFETY: Every `(y, x)` in `(0..height) x (0..width)` was initialized above.
        unsafe { dst.finish_into(image, Plane::A, max_channel) }
    }
}

pub(crate) fn import_alpha_uninit_with_threads(
    rgb: &rgb::Image,
    image: &mut image::Image,
    threads: u32,
) -> AvifResult<()> {
    validate_alpha_import(rgb, image)?;
    let width = usize_from_u32(image.width)?;
    let height_usize = usize_from_u32(image.height)?;
    let min_pixels = if rgb.depth == 8 && image.depth == 8 {
        MIN_MT_PIXELS_ALPHA_INT8
    } else {
        MIN_MT_PIXELS_RGB_TO_YUV
    };
    let row_worker_cap = height_usize / MIN_MT_ROWS_PER_WORKER;
    let workers =
        if threads > 1 && width.saturating_mul(height_usize) >= min_pixels && row_worker_cap >= 2 {
            (threads as usize)
                .min(MAX_RGB_TO_YUV_THREADS)
                .min(row_worker_cap)
        } else {
            1
        };
    if workers <= 1 {
        return import_alpha_uninit(rgb, image);
    }
    let src_alpha_offset = rgb.format.alpha_offset();
    let channel_count = rgb.format.channel_count() as usize;
    let max_channel = image.max_channel();
    let src_max_channel_f = rgb.max_channel_f();

    let src_row_bytes = usize_from_u32(rgb.row_bytes)?;
    let src_req_elems = checked_mul!(width, channel_count)?;
    let src_depth = rgb.depth;
    let image_depth = image.depth;
    let fast_4ch = matches!(
        rgb.format,
        Format::Rgba | Format::Bgra | Format::Argb | Format::Abgr
    );
    let (src_all8, src_all16, src_stride_elems): (&[u8], &[u16], usize) = if src_depth > 8 {
        let stride_u16 = src_row_bytes / 2;
        let total_u16 = checked_add!(checked_mul!(height_usize - 1, stride_u16)?, src_req_elems)?;
        let slice16 = rgb
            .pixels
            .as_ref()
            .ok_or(AvifError::NoContent)?
            .slice16(0, u32_from_usize(total_u16)?)?;
        (&[], slice16, stride_u16)
    } else {
        let total_u8 = checked_add!(
            checked_mul!(height_usize - 1, src_row_bytes)?,
            src_req_elems
        )?;
        let slice8 = rgb
            .pixels
            .as_ref()
            .ok_or(AvifError::NoContent)?
            .slice(0, u32_from_usize(total_u8)?)?;
        (slice8, &[], src_row_bytes)
    };
    let rows_per = height_usize / workers;
    let extra = height_usize % workers;

    if image_depth == 8 {
        let mut dst = UninitPlane8::for_image(image, Plane::A)?;
        let dst_stride = dst.stride_elems();
        let spare = dst.spare_all_mut()?;
        let mut stripes = create_vec_exact(workers)?;
        let mut rem = spare;
        let mut row_start = 0usize;
        for idx in 0..workers {
            let row_count = rows_per + usize::from(idx < extra);
            let split_at = row_count * dst_stride;
            let (head, tail) = rem.split_at_mut(split_at);
            rem = tail;
            stripes.try_push((row_start, row_count, head))?;
            row_start += row_count;
        }
        let process_stripe8 =
            |(r0, r_cnt, dst_chunk): (usize, usize, &mut [std::mem::MaybeUninit<u8>])| {
                if src_depth == 8 && fast_4ch {
                    extract_alpha_4ch_stripe_u8(
                        src_all8,
                        src_stride_elems,
                        r0,
                        r_cnt,
                        dst_chunk,
                        dst_stride,
                        width,
                        src_alpha_offset,
                    );
                    return;
                }
                for local_y in 0..r_cnt {
                    let y = r0 + local_y;
                    let dst_off = local_y * dst_stride;
                    let dst_row = &mut dst_chunk[dst_off..dst_off + width];
                    let src_off = y * src_stride_elems;
                    if src_depth == 8 {
                        let src_row = &src_all8[src_off..src_off + src_req_elems];
                        for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                            cell.write(src_row[(x * channel_count) + src_alpha_offset]);
                        }
                    } else {
                        let src_row = &src_all16[src_off..src_off + src_req_elems];
                        for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                            let v = src_row[(x * channel_count) + src_alpha_offset];
                            cell.write(rgb::Image::rescale_alpha_value(
                                v,
                                src_max_channel_f,
                                max_channel,
                            ) as u8);
                        }
                    }
                }
            };
        let (first, rest) = stripes.split_first_mut().ok_or(AvifError::ReformatFailed)?;
        std::thread::scope(|scope| -> AvifResult<()> {
            let mut handles = create_vec_exact(rest.len())?;
            for stripe in rest.iter_mut() {
                let (r0, r_cnt, dst_chunk) = (stripe.0, stripe.1, &mut *stripe.2);
                let handle = std::thread::Builder::new()
                    .spawn_scoped(scope, move || process_stripe8((r0, r_cnt, dst_chunk)))
                    .map_err(|_| AvifError::ReformatFailed)?;
                handles.try_push(handle)?;
            }
            process_stripe8((first.0, first.1, &mut *first.2));
            let mut join_ok = true;
            for handle in handles {
                if handle.join().is_err() {
                    join_ok = false;
                }
            }
            if !join_ok {
                return AvifError::reformat_failed();
            }
            Ok(())
        })?;
        // SAFETY: Every stripe covering `(0..height) x (0..width)` initialized its cells.
        unsafe { dst.finish_into(image, Plane::A, max_channel) }
    } else {
        let mut dst = UninitPlane16::for_image(image, Plane::A)?;
        let dst_stride = dst.stride_elems();
        let spare = dst.spare_all_mut()?;
        let mut stripes = create_vec_exact(workers)?;
        let mut rem = spare;
        let mut row_start = 0usize;
        for idx in 0..workers {
            let row_count = rows_per + usize::from(idx < extra);
            let split_at = row_count * dst_stride;
            let (head, tail) = rem.split_at_mut(split_at);
            rem = tail;
            stripes.try_push((row_start, row_count, head))?;
            row_start += row_count;
        }
        let same_depth = src_depth == image_depth;
        let process_stripe16 =
            |(r0, r_cnt, dst_chunk): (usize, usize, &mut [std::mem::MaybeUninit<u16>])| {
                for local_y in 0..r_cnt {
                    let y = r0 + local_y;
                    let dst_off = local_y * dst_stride;
                    let dst_row = &mut dst_chunk[dst_off..dst_off + width];
                    let src_off = y * src_stride_elems;
                    if src_depth > 8 {
                        let src_row = &src_all16[src_off..src_off + src_req_elems];
                        if same_depth {
                            for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                                cell.write(src_row[(x * channel_count) + src_alpha_offset]);
                            }
                        } else {
                            for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                                let v = src_row[(x * channel_count) + src_alpha_offset];
                                cell.write(rgb::Image::rescale_alpha_value(
                                    v,
                                    src_max_channel_f,
                                    max_channel,
                                ));
                            }
                        }
                    } else {
                        let src_row = &src_all8[src_off..src_off + src_req_elems];
                        for (x, cell) in dst_row.iter_mut().enumerate().take(width) {
                            let v = src_row[(x * channel_count) + src_alpha_offset] as u16;
                            cell.write(rgb::Image::rescale_alpha_value(
                                v,
                                src_max_channel_f,
                                max_channel,
                            ));
                        }
                    }
                }
            };
        let (first, rest) = stripes.split_first_mut().ok_or(AvifError::ReformatFailed)?;
        std::thread::scope(|scope| -> AvifResult<()> {
            let mut handles = create_vec_exact(rest.len())?;
            for stripe in rest.iter_mut() {
                let (r0, r_cnt, dst_chunk) = (stripe.0, stripe.1, &mut *stripe.2);
                let handle = std::thread::Builder::new()
                    .spawn_scoped(scope, move || process_stripe16((r0, r_cnt, dst_chunk)))
                    .map_err(|_| AvifError::ReformatFailed)?;
                handles.try_push(handle)?;
            }
            process_stripe16((first.0, first.1, &mut *first.2));
            let mut join_ok = true;
            for handle in handles {
                if handle.join().is_err() {
                    join_ok = false;
                }
            }
            if !join_ok {
                return AvifError::reformat_failed();
            }
            Ok(())
        })?;
        // SAFETY: Every stripe covering `(0..height) x (0..width)` initialized its cells.
        unsafe { dst.finish_into(image, Plane::A, max_channel) }
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_alpha_4ch_stripe_u8(
    src_all: &[u8],
    src_stride: usize,
    r0: usize,
    r_cnt: usize,
    dst_chunk: &mut [std::mem::MaybeUninit<u8>],
    dst_stride: usize,
    width: usize,
    alpha_offset: usize,
) {
    assert!(alpha_offset < 4);
    #[cfg(target_arch = "x86_64")]
    {
        if width >= 32 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime check verified `avx2` support, and `alpha_offset < 4`.
            unsafe {
                extract_alpha_4ch_stripe_avx2(
                    src_all,
                    src_stride,
                    r0,
                    r_cnt,
                    dst_chunk,
                    dst_stride,
                    width,
                    alpha_offset,
                );
            }
            return;
        }
    }
    let src_req = width * 4;
    for local_y in 0..r_cnt {
        let src_off = (r0 + local_y) * src_stride;
        let dst_off = local_y * dst_stride;
        extract_alpha_4ch_row_scalar(
            &src_all[src_off..src_off + src_req],
            &mut dst_chunk[dst_off..dst_off + width],
            alpha_offset,
        );
    }
}

#[inline]
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn extract_alpha_4ch_row_scalar(
    src: &[u8],
    dst: &mut [std::mem::MaybeUninit<u8>],
    alpha_offset: usize,
) {
    assert!(alpha_offset < 4 && src.len() >= dst.len() * 4);
    for (cell, px) in dst.iter_mut().zip(src.chunks_exact(4)) {
        cell.write(px[alpha_offset]);
    }
}

/// # Safety
/// Caller must ensure runtime `avx2` support, `alpha_offset < 4`, `src_chunk.len() >= 128`,
/// and `dst_ptr` is valid for 32 writable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extract_alpha_4ch_block32_avx2(
    src_chunk: &[u8],
    dst_ptr: *mut u8,
    alpha_offset: usize,
    shuf: std::arch::x86_64::__m256i,
    perm: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;
    let p = src_chunk.as_ptr();
    // SAFETY: Caller guarantees `src_chunk` has at least 128 readable bytes.
    let (r0, r1, r2, r3) = unsafe {
        (
            _mm256_loadu_si256(p.cast()),
            _mm256_loadu_si256(p.add(32).cast()),
            _mm256_loadu_si256(p.add(64).cast()),
            _mm256_loadu_si256(p.add(96).cast()),
        )
    };
    let (v0, v1, v2, v3) = if alpha_offset == 3 {
        (
            _mm256_srli_epi32::<24>(r0),
            _mm256_srli_epi32::<24>(r1),
            _mm256_srli_epi32::<24>(r2),
            _mm256_srli_epi32::<24>(r3),
        )
    } else {
        (
            _mm256_shuffle_epi8(r0, shuf),
            _mm256_shuffle_epi8(r1, shuf),
            _mm256_shuffle_epi8(r2, shuf),
            _mm256_shuffle_epi8(r3, shuf),
        )
    };
    let q0 = _mm256_packs_epi32(v0, v1);
    let q1 = _mm256_packs_epi32(v2, v3);
    let ordered = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(q0, q1), perm);
    // SAFETY: Caller guarantees `dst_ptr` points to 32 writable bytes.
    unsafe { _mm256_storeu_si256(dst_ptr.cast(), ordered) };
}

/// # Safety
/// Caller must ensure runtime `avx2` support and `alpha_offset < 4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn alpha_shuf_and_perm(
    alpha_offset: usize,
) -> (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i) {
    use std::arch::x86_64::*;
    let off = alpha_offset as i8;
    let shuf = _mm256_setr_epi8(
        off,
        -1,
        -1,
        -1,
        off + 4,
        -1,
        -1,
        -1,
        off + 8,
        -1,
        -1,
        -1,
        off + 12,
        -1,
        -1,
        -1,
        off,
        -1,
        -1,
        -1,
        off + 4,
        -1,
        -1,
        -1,
        off + 8,
        -1,
        -1,
        -1,
        off + 12,
        -1,
        -1,
        -1,
    );
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    (shuf, perm)
}

/// # Safety
/// Caller must ensure runtime `avx2` support and `alpha_offset < 4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    unknown_lints,
    clippy::chunks_exact_to_as_chunks
)]
unsafe fn extract_alpha_4ch_stripe_avx2(
    src_all: &[u8],
    src_stride: usize,
    r0: usize,
    r_cnt: usize,
    dst_chunk: &mut [std::mem::MaybeUninit<u8>],
    dst_stride: usize,
    width: usize,
    alpha_offset: usize,
) {
    assert!(alpha_offset < 4);
    // SAFETY: `avx2` is enabled on this function and `alpha_offset < 4`.
    let (shuf, perm) = unsafe { alpha_shuf_and_perm(alpha_offset) };
    let src_req = width * 4;
    for local_y in 0..r_cnt {
        let src_off = (r0 + local_y) * src_stride;
        let dst_off = local_y * dst_stride;
        let src = &src_all[src_off..src_off + src_req];
        let dst = &mut dst_chunk[dst_off..dst_off + width];
        let mut processed = 0usize;
        for (s_blk, d_blk) in src.chunks_exact(128).zip(dst.chunks_exact_mut(32)) {
            // SAFETY: `s_blk` is 128 bytes and `d_blk` is 32 `MaybeUninit<u8>` elements.
            unsafe {
                extract_alpha_4ch_block32_avx2(
                    s_blk,
                    d_blk.as_mut_ptr().cast::<u8>(),
                    alpha_offset,
                    shuf,
                    perm,
                );
            }
            processed += 32;
        }
        if let (Some(rs), Some(rd)) = (src.get(processed * 4..), dst.get_mut(processed..)) {
            extract_alpha_4ch_row_scalar(rs, rd, alpha_offset);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::pixels::Pixels;

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
                yuv_range: YuvRange::Limited,
                ..Default::default()
            };
            assert!(yuv.allocate_planes(Category::Color).is_ok());
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
                    let i = (x * dst.pixel_size()) as usize;
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

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn next_u16_bounded(&mut self, max_inclusive: u16) -> u16 {
            let span = (max_inclusive as u64) + 1;
            (self.next_u64() % span) as u16
        }
    }

    fn ref_quant_u8_floor(bias: f32, range: f32, v: f32) -> u8 {
        clamp_i32((0.5 + (v * range + bias)).floor() as i32, 0, 255) as u8
    }

    fn ref_quant_u16_floor(bias: f32, range: f32, max_ch: u16, v: f32) -> u16 {
        clamp_i32((0.5 + (v * range + bias)).floor() as i32, 0, max_ch as i32) as u16
    }

    fn ref_rgb_px_to_yuv_floor(
        mode: Mode,
        r: f32,
        g: f32,
        b: f32,
        rgb_max_f: f32,
        range_y: f32,
        range_uv: f32,
    ) -> (f32, f32, f32) {
        match mode {
            Mode::YuvCoefficients(kr, kg, kb) => {
                let y = (kr * r) + (kg * g) + (kb * b);
                (
                    y,
                    (b - y) / (2.0 * (1.0 - kb)),
                    (r - y) / (2.0 * (1.0 - kr)),
                )
            }
            Mode::Identity => (g, b, r),
            Mode::Ycgco => (
                0.5 * g + 0.25 * (r + b),
                0.5 * g - 0.25 * (r + b),
                0.5 * (r - b),
            ),
            Mode::YcgcoRe | Mode::YcgcoRo => {
                let ri = ((r * rgb_max_f).clamp(0.0, rgb_max_f) + 0.5).floor() as i32;
                let gi = ((g * rgb_max_f).clamp(0.0, rgb_max_f) + 0.5).floor() as i32;
                let bi = ((b * rgb_max_f).clamp(0.0, rgb_max_f) + 0.5).floor() as i32;
                let co = ri - bi;
                let t = bi + (co >> 1);
                let cg = gi - t;
                (
                    (t + (cg >> 1)) as f32 / range_y,
                    cg as f32 / range_uv,
                    co as f32 / range_uv,
                )
            }
        }
    }

    fn ref_read_rgba(rgb: &rgb::Image, x: usize, y: u32) -> (f32, f32, f32, u16) {
        let ch = rgb.channel_count() as usize;
        let ro = rgb.format.r_offset();
        let go = rgb.format.g_offset();
        let bo = rgb.format.b_offset();
        let ao = rgb.format.alpha_offset();
        let base = x * ch;
        if rgb.depth > 8 {
            let max_f = rgb.max_channel_f();
            let row = rgb.row16(y).unwrap();
            (
                row[base + ro] as f32 / max_f,
                row[base + go] as f32 / max_f,
                row[base + bo] as f32 / max_f,
                row[base + ao],
            )
        } else {
            let row = rgb.row(y).unwrap();
            (
                row[base + ro] as f32 / 255.0,
                row[base + go] as f32 / 255.0,
                row[base + bo] as f32 / 255.0,
                row[base + ao] as u16,
            )
        }
    }

    fn ref_store_sample(image: &mut image::Image, plane: Plane, x: usize, y: u32, v: u16) {
        if image.depth > 8 {
            image.row16_mut(plane, y).unwrap()[x] = v;
        } else {
            image.row_mut(plane, y).unwrap()[x] = v as u8;
        }
    }

    // Standalone reference scalar converter using `.floor() as i32` throughout.
    fn ref_rgb_to_yuv_floor(rgb: &rgb::Image, image: &mut image::Image, import_all_alpha: bool) {
        let mode: Mode = (&*image).into();
        let (bias_y, range_y) = bias_and_range_y(image);
        let (bias_uv, range_uv) = if mode == Mode::Identity {
            (bias_y, range_y)
        } else {
            bias_and_range_uv(image)
        };
        let uv_range_for_px = if image.yuv_format == PixelFormat::Yuv400 { 0.0 } else { range_uv };
        let rgb_max_f = rgb.max_channel_f();
        let yuv_max = image.max_channel();
        let w = image.width as usize;
        let h = image.height;
        let q = |bias: f32, range: f32, val: f32, depth: u8| -> u16 {
            if depth > 8 {
                ref_quant_u16_floor(bias, range, yuv_max, val)
            } else {
                ref_quant_u8_floor(bias, range, val) as u16
            }
        };
        let depth = image.depth;
        match image.yuv_format {
            PixelFormat::Yuv444 => {
                for y in 0..h {
                    for x in 0..w {
                        let (r, g, b, _) = ref_read_rgba(rgb, x, y);
                        let p = ref_rgb_px_to_yuv_floor(
                            mode,
                            r,
                            g,
                            b,
                            rgb_max_f,
                            range_y,
                            uv_range_for_px,
                        );
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p.0, depth));
                        ref_store_sample(image, Plane::U, x, y, q(bias_uv, range_uv, p.1, depth));
                        ref_store_sample(image, Plane::V, x, y, q(bias_uv, range_uv, p.2, depth));
                    }
                }
            }
            PixelFormat::Yuv400 => {
                for y in 0..h {
                    for x in 0..w {
                        let (r, g, b, _) = ref_read_rgba(rgb, x, y);
                        let p = ref_rgb_px_to_yuv_floor(
                            mode,
                            r,
                            g,
                            b,
                            rgb_max_f,
                            range_y,
                            uv_range_for_px,
                        );
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p.0, depth));
                    }
                }
            }
            PixelFormat::Yuv422 => {
                for y in 0..h {
                    for x in (0..w.saturating_sub(1)).step_by(2) {
                        let (r0, g0, b0, _) = ref_read_rgba(rgb, x, y);
                        let (r1, g1, b1, _) = ref_read_rgba(rgb, x + 1, y);
                        let p0 =
                            ref_rgb_px_to_yuv_floor(mode, r0, g0, b0, rgb_max_f, range_y, range_uv);
                        let p1 =
                            ref_rgb_px_to_yuv_floor(mode, r1, g1, b1, rgb_max_f, range_y, range_uv);
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p0.0, depth));
                        ref_store_sample(
                            image,
                            Plane::Y,
                            x + 1,
                            y,
                            q(bias_y, range_y, p1.0, depth),
                        );
                        let u = q(bias_uv, range_uv, (p0.1 + p1.1) / 2.0, depth);
                        let v = q(bias_uv, range_uv, (p0.2 + p1.2) / 2.0, depth);
                        ref_store_sample(image, Plane::U, x >> 1, y, u);
                        ref_store_sample(image, Plane::V, x >> 1, y, v);
                    }
                    if w % 2 != 0 {
                        let x = w - 1;
                        let (r, g, b, _) = ref_read_rgba(rgb, x, y);
                        let p =
                            ref_rgb_px_to_yuv_floor(mode, r, g, b, rgb_max_f, range_y, range_uv);
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p.0, depth));
                        ref_store_sample(
                            image,
                            Plane::U,
                            x >> 1,
                            y,
                            q(bias_uv, range_uv, p.1, depth),
                        );
                        ref_store_sample(
                            image,
                            Plane::V,
                            x >> 1,
                            y,
                            q(bias_uv, range_uv, p.2, depth),
                        );
                    }
                }
            }
            PixelFormat::Yuv420 => {
                for y in (0..h.saturating_sub(1)).step_by(2) {
                    let uv_y = y >> 1;
                    for x in (0..w.saturating_sub(1)).step_by(2) {
                        let (r00, g00, b00, _) = ref_read_rgba(rgb, x, y);
                        let (r01, g01, b01, _) = ref_read_rgba(rgb, x + 1, y);
                        let (r10, g10, b10, _) = ref_read_rgba(rgb, x, y + 1);
                        let (r11, g11, b11, _) = ref_read_rgba(rgb, x + 1, y + 1);
                        let p00 = ref_rgb_px_to_yuv_floor(
                            mode, r00, g00, b00, rgb_max_f, range_y, range_uv,
                        );
                        let p01 = ref_rgb_px_to_yuv_floor(
                            mode, r01, g01, b01, rgb_max_f, range_y, range_uv,
                        );
                        let p10 = ref_rgb_px_to_yuv_floor(
                            mode, r10, g10, b10, rgb_max_f, range_y, range_uv,
                        );
                        let p11 = ref_rgb_px_to_yuv_floor(
                            mode, r11, g11, b11, rgb_max_f, range_y, range_uv,
                        );
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p00.0, depth));
                        ref_store_sample(
                            image,
                            Plane::Y,
                            x + 1,
                            y,
                            q(bias_y, range_y, p01.0, depth),
                        );
                        ref_store_sample(
                            image,
                            Plane::Y,
                            x,
                            y + 1,
                            q(bias_y, range_y, p10.0, depth),
                        );
                        ref_store_sample(
                            image,
                            Plane::Y,
                            x + 1,
                            y + 1,
                            q(bias_y, range_y, p11.0, depth),
                        );
                        let avg_u = (p00.1 + p01.1 + p10.1 + p11.1) / 4.0;
                        let avg_v = (p00.2 + p01.2 + p10.2 + p11.2) / 4.0;
                        ref_store_sample(
                            image,
                            Plane::U,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, avg_u, depth),
                        );
                        ref_store_sample(
                            image,
                            Plane::V,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, avg_v, depth),
                        );
                    }
                    if w % 2 != 0 {
                        let x = w - 1;
                        let (r0, g0, b0, _) = ref_read_rgba(rgb, x, y);
                        let (r1, g1, b1, _) = ref_read_rgba(rgb, x, y + 1);
                        let p0 =
                            ref_rgb_px_to_yuv_floor(mode, r0, g0, b0, rgb_max_f, range_y, range_uv);
                        let p1 =
                            ref_rgb_px_to_yuv_floor(mode, r1, g1, b1, rgb_max_f, range_y, range_uv);
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p0.0, depth));
                        ref_store_sample(
                            image,
                            Plane::Y,
                            x,
                            y + 1,
                            q(bias_y, range_y, p1.0, depth),
                        );
                        let avg_u = (p0.1 + p1.1) / 2.0;
                        let avg_v = (p0.2 + p1.2) / 2.0;
                        ref_store_sample(
                            image,
                            Plane::U,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, avg_u, depth),
                        );
                        ref_store_sample(
                            image,
                            Plane::V,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, avg_v, depth),
                        );
                    }
                }
                if h % 2 != 0 {
                    let y = h - 1;
                    let uv_y = y >> 1;
                    for x in (0..w.saturating_sub(1)).step_by(2) {
                        let (r0, g0, b0, _) = ref_read_rgba(rgb, x, y);
                        let (r1, g1, b1, _) = ref_read_rgba(rgb, x + 1, y);
                        let p0 =
                            ref_rgb_px_to_yuv_floor(mode, r0, g0, b0, rgb_max_f, range_y, range_uv);
                        let p1 =
                            ref_rgb_px_to_yuv_floor(mode, r1, g1, b1, rgb_max_f, range_y, range_uv);
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p0.0, depth));
                        ref_store_sample(
                            image,
                            Plane::Y,
                            x + 1,
                            y,
                            q(bias_y, range_y, p1.0, depth),
                        );
                        let avg_u = (p0.1 + p1.1) / 2.0;
                        let avg_v = (p0.2 + p1.2) / 2.0;
                        ref_store_sample(
                            image,
                            Plane::U,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, avg_u, depth),
                        );
                        ref_store_sample(
                            image,
                            Plane::V,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, avg_v, depth),
                        );
                    }
                    if w % 2 != 0 {
                        let x = w - 1;
                        let (r, g, b, _) = ref_read_rgba(rgb, x, y);
                        let p =
                            ref_rgb_px_to_yuv_floor(mode, r, g, b, rgb_max_f, range_y, range_uv);
                        ref_store_sample(image, Plane::Y, x, y, q(bias_y, range_y, p.0, depth));
                        ref_store_sample(
                            image,
                            Plane::U,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, p.1, depth),
                        );
                        ref_store_sample(
                            image,
                            Plane::V,
                            x >> 1,
                            uv_y,
                            q(bias_uv, range_uv, p.2, depth),
                        );
                    }
                }
            }
            _ => unreachable!(),
        }
        if image.has_plane(Plane::A) {
            if rgb.has_alpha() {
                if rgb.depth == image.depth {
                    for y in 0..h {
                        for x in 0..w {
                            let (_, _, _, a) = ref_read_rgba(rgb, x, y);
                            ref_store_sample(image, Plane::A, x, y, a);
                        }
                    }
                } else if import_all_alpha {
                    for y in 0..h {
                        for x in 0..w {
                            let (_, _, _, a) = ref_read_rgba(rgb, x, y);
                            let alpha_f = (a as f32) / rgb_max_f;
                            let scaled = ((0.5 + (alpha_f * (yuv_max as f32))) as u16).min(yuv_max);
                            ref_store_sample(image, Plane::A, x, y, scaled);
                        }
                    }
                }
            } else if import_all_alpha {
                for y in 0..h {
                    for x in 0..w {
                        ref_store_sample(image, Plane::A, x, y, yuv_max);
                    }
                }
            }
        }
    }

    fn alloc_sentinel_yuv(
        width: u32,
        height: u32,
        depth: u8,
        yuv_format: PixelFormat,
        yuv_range: YuvRange,
        mc: MatrixCoefficients,
        with_alpha: bool,
    ) -> image::Image {
        let mut img = image::Image {
            width,
            height,
            depth,
            yuv_format,
            yuv_range,
            color_primaries: ColorPrimaries::Srgb,
            matrix_coefficients: mc,
            ..Default::default()
        };
        img.allocate_planes(Category::Color).unwrap();
        if with_alpha {
            img.allocate_planes(Category::Alpha).unwrap();
        }
        for plane in image::ALL_PLANES {
            if !img.has_plane(plane) {
                continue;
            }
            let ph = img.height(plane) as u32;
            for y in 0..ph {
                if depth > 8 {
                    img.row16_mut(plane, y).unwrap().fill(0xBEEF);
                } else {
                    img.row_mut(plane, y).unwrap().fill(0xA5);
                }
            }
        }
        img
    }

    fn assert_images_bit_exact(actual: &image::Image, expected: &image::Image, ctx: &str) {
        for plane in image::ALL_PLANES {
            assert_eq!(
                actual.has_plane(plane),
                expected.has_plane(plane),
                "has_plane({plane:?}) mismatch: {ctx}"
            );
            if !expected.has_plane(plane) {
                continue;
            }
            let ph = expected.height(plane) as u32;
            for y in 0..ph {
                if expected.depth > 8 {
                    let a = actual.row16(plane, y).unwrap();
                    let e = expected.row16(plane, y).unwrap();
                    assert_eq!(a, e, "plane {plane:?} row {y} mismatch: {ctx}");
                } else {
                    let a = actual.row(plane, y).unwrap();
                    let e = expected.row(plane, y).unwrap();
                    assert_eq!(a, e, "plane {plane:?} row {y} mismatch: {ctx}");
                }
            }
        }
    }

    fn make_rgb_from_rgba_fn(
        width: u32,
        height: u32,
        depth: u8,
        format: Format,
        extra_row_pad_elems: usize,
        mut f: impl FnMut(u32, u32) -> [u16; 4],
    ) -> rgb::Image {
        let ch = format.channel_count() as usize;
        let ro = format.r_offset();
        let go = format.g_offset();
        let bo = format.b_offset();
        let ao = format.alpha_offset();
        let has_a = format.has_alpha();
        let row_elems = (width as usize) * ch + extra_row_pad_elems;
        if depth > 8 {
            let mut buf = vec![0xDEADu16; row_elems * (height as usize)];
            for y in 0..height {
                let row = &mut buf[(y as usize) * row_elems..(y as usize + 1) * row_elems];
                for x in 0..width {
                    let [r, g, b, a] = f(x, y);
                    let base = (x as usize) * ch;
                    row[base + ro] = r;
                    row[base + go] = g;
                    row[base + bo] = b;
                    if has_a {
                        row[base + ao] = a;
                    }
                }
            }
            rgb::Image {
                width,
                height,
                depth,
                format,
                pixels: Some(Pixels::Buffer16(buf)),
                row_bytes: (row_elems * 2) as u32,
                ..Default::default()
            }
        } else {
            let mut buf = vec![0x5Au8; row_elems * (height as usize)];
            for y in 0..height {
                let row = &mut buf[(y as usize) * row_elems..(y as usize + 1) * row_elems];
                for x in 0..width {
                    let [r, g, b, a] = f(x, y);
                    let base = (x as usize) * ch;
                    row[base + ro] = r as u8;
                    row[base + go] = g as u8;
                    row[base + bo] = b as u8;
                    if has_a {
                        row[base + ao] = a as u8;
                    }
                }
            }
            rgb::Image {
                width,
                height,
                depth,
                format,
                pixels: Some(Pixels::Buffer(buf)),
                row_bytes: row_elems as u32,
                ..Default::default()
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_all_paths_0ulp(
        rgb: &rgb::Image,
        yuv_depth: u8,
        pf: PixelFormat,
        range: YuvRange,
        mc: MatrixCoefficients,
        alloc_alpha: bool,
        ctx: &str,
    ) {
        let expect_fuse = alloc_alpha && rgb.has_alpha();

        let mut ref_direct =
            alloc_sentinel_yuv(rgb.width, rgb.height, yuv_depth, pf, range, mc, alloc_alpha);
        ref_rgb_to_yuv_floor(rgb, &mut ref_direct, expect_fuse);

        // 1. AVX2 uninit-plane path (including fused alpha return value).
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            let mut got_avx2 = image::Image {
                width: rgb.width,
                height: rgb.height,
                depth: yuv_depth,
                yuv_format: pf,
                yuv_range: range,
                color_primaries: ColorPrimaries::Srgb,
                matrix_coefficients: mc,
                ..Default::default()
            };
            let fused =
                avx2::try_rgb_to_yuv_uninit_with_threads(rgb, &mut got_avx2, alloc_alpha, 1)
                    .unwrap();
            assert_eq!(fused, Some(expect_fuse), "avx2 fused flag: {ctx}");
            if expect_fuse || !alloc_alpha {
                assert_images_bit_exact(&got_avx2, &ref_direct, &format!("avx2 {ctx}"));
            } else {
                let mut ref_no_a =
                    alloc_sentinel_yuv(rgb.width, rgb.height, yuv_depth, pf, range, mc, false);
                ref_rgb_to_yuv_floor(rgb, &mut ref_no_a, false);
                assert_images_bit_exact(&got_avx2, &ref_no_a, &format!("avx2 {ctx}"));
            }
        }

        // 2. Dispatched `rgb_to_yuv` entrypoint & non-x86_64 `rgb_to_yuv_scalar_fallback`.
        let mut got_disp =
            alloc_sentinel_yuv(rgb.width, rgb.height, yuv_depth, pf, range, mc, alloc_alpha);
        let disp_fused = rgb_to_yuv(rgb, &mut got_disp).unwrap();
        let mut got_fallback = image::Image {
            width: rgb.width,
            height: rgb.height,
            depth: yuv_depth,
            yuv_format: pf,
            yuv_range: range,
            color_primaries: ColorPrimaries::Srgb,
            matrix_coefficients: mc,
            ..Default::default()
        };
        let fb_fused = rgb_to_yuv_scalar_fallback(rgb, &mut got_fallback, alloc_alpha).unwrap();
        assert_eq!(
            fb_fused,
            pf == PixelFormat::Yuv444 && expect_fuse,
            "fallback fused flag: {ctx}"
        );
        if fb_fused {
            assert_images_bit_exact(&got_fallback, &ref_direct, &format!("fallback {ctx}"));
        }
        if pf == PixelFormat::Yuv444 {
            assert_eq!(disp_fused, expect_fuse, "dispatch 444 fused flag: {ctx}");
        }
        if disp_fused {
            assert_images_bit_exact(&got_disp, &ref_direct, &format!("disp {ctx}"));
        } else {
            let mut ref_no_a =
                alloc_sentinel_yuv(rgb.width, rgb.height, yuv_depth, pf, range, mc, false);
            ref_rgb_to_yuv_floor(rgb, &mut ref_no_a, false);
            for plane in image::YUV_PLANES {
                if !ref_no_a.has_plane(plane) {
                    continue;
                }
                for y in 0..ref_no_a.height(plane) as u32 {
                    if yuv_depth > 8 {
                        assert_eq!(
                            got_disp.row16(plane, y).unwrap(),
                            ref_no_a.row16(plane, y).unwrap(),
                            "disp {plane:?} row {y}: {ctx}"
                        );
                    } else {
                        assert_eq!(
                            got_disp.row(plane, y).unwrap(),
                            ref_no_a.row(plane, y).unwrap(),
                            "disp {plane:?} row {y}: {ctx}"
                        );
                    }
                }
            }
        }

        // 3. Scalar path (verifies `.floor()` elimination + scalar 444 fused alpha).
        let expect_scalar_fuse = pf == PixelFormat::Yuv444 && expect_fuse;
        let mut got_scalar =
            alloc_sentinel_yuv(rgb.width, rgb.height, yuv_depth, pf, range, mc, alloc_alpha);
        let sc_fused = rgb_to_yuv_scalar(rgb, &mut got_scalar).unwrap();
        assert_eq!(sc_fused, expect_scalar_fuse, "scalar fused flag: {ctx}");
        if expect_scalar_fuse {
            assert_images_bit_exact(&got_scalar, &ref_direct, &format!("scalar {ctx}"));
        } else {
            let mut ref_unfused =
                alloc_sentinel_yuv(rgb.width, rgb.height, yuv_depth, pf, range, mc, alloc_alpha);
            // Leave Plane::A filled with sentinel when scalar does not fuse alpha (420/422/400).
            let mut tmp =
                alloc_sentinel_yuv(rgb.width, rgb.height, yuv_depth, pf, range, mc, false);
            ref_rgb_to_yuv_floor(rgb, &mut tmp, false);
            for plane in image::YUV_PLANES {
                if !tmp.has_plane(plane) {
                    continue;
                }
                for y in 0..tmp.height(plane) as u32 {
                    if yuv_depth > 8 {
                        let src_r = tmp.row16(plane, y).unwrap();
                        ref_unfused
                            .row16_mut(plane, y)
                            .unwrap()
                            .copy_from_slice(src_r);
                    } else {
                        let src_r = tmp.row(plane, y).unwrap();
                        ref_unfused
                            .row_mut(plane, y)
                            .unwrap()
                            .copy_from_slice(src_r);
                    }
                }
            }
            assert_images_bit_exact(&got_scalar, &ref_unfused, &format!("scalar {ctx}"));
        }
    }

    const ALL_6_FORMATS: [Format; 6] = [
        Format::Rgb,
        Format::Bgr,
        Format::Rgba,
        Format::Bgra,
        Format::Argb,
        Format::Abgr,
    ];

    const ALL_PFS: [PixelFormat; 4] = [
        PixelFormat::Yuv444,
        PixelFormat::Yuv422,
        PixelFormat::Yuv420,
        PixelFormat::Yuv400,
    ];

    const ALL_MCS: [MatrixCoefficients; 7] = [
        MatrixCoefficients::Bt601,
        MatrixCoefficients::Bt709,
        MatrixCoefficients::Bt2020Ncl,
        MatrixCoefficients::Fcc,
        MatrixCoefficients::Identity,
        MatrixCoefficients::Ycgco,
        MatrixCoefficients::YcgcoRe,
    ];

    #[test]
    fn rgb_to_yuv_avx2_all_256_ramps_and_corners_0ulp() {
        // Pack all 256 single-channel ramps (R, G, B, gray, complement) + all 8^3 = 512 corner
        // triples from [0, 1, 2, 127, 128, 253, 254, 255] into a 64x29 image = 1856 pixels.
        let corners8: [u16; 8] = [0, 1, 2, 127, 128, 253, 254, 255];
        let mut samples = Vec::with_capacity(256 * 5 + 512);
        for v in 0u16..=255 {
            samples.push([v, 0, 0, v]);
            samples.push([0, v, 0, 255 - v]);
            samples.push([0, 0, v, v ^ 0x55]);
            samples.push([v, v, v, v ^ 0xAA]);
            samples.push([v, 255 - v, v ^ 0x55, (v.wrapping_mul(7)) & 0xFF]);
        }
        for &r in &corners8 {
            for &g in &corners8 {
                for &b in &corners8 {
                    samples.push([r, g, b, (r ^ g ^ b) & 0xFF]);
                }
            }
        }
        let w = 64u32;
        let h = (samples.len() as u32).div_ceil(w);
        for fmt in ALL_6_FORMATS {
            let rgb = make_rgb_from_rgba_fn(w, h, 8, fmt, 0, |x, y| {
                let idx = (y * w + x) as usize;
                samples.get(idx).copied().unwrap_or([0, 128, 255, 255])
            });
            for mc in ALL_MCS {
                let out_depths: &[u8] =
                    if mc == MatrixCoefficients::YcgcoRe { &[10] } else { &[8] };
                for &od in out_depths {
                    for range in [YuvRange::Full, YuvRange::Limited] {
                        for pf in ALL_PFS {
                            verify_all_paths_0ulp(
                                &rgb,
                                od,
                                pf,
                                range,
                                mc,
                                fmt.has_alpha(),
                                &format!("ramp {fmt:?} {mc:?} d{od} {range:?} {pf:?}"),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rgb_to_yuv_avx2_widths_heights_formats_0ulp() {
        let widths = [1u32, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 49];
        let heights = [1u32, 2, 3, 5];
        let mut rng = SplitMix64(0xC0FF_EE12_3456_789A);

        // Tight unpadded (extra_row_pad_elems = 0) + padded rows across all widths x heights x
        // formats x subsamplings x ranges at 8-bit and 16-bit (catches any 3-channel overread or
        // row-tail bug!).
        for &w in &widths {
            for &h in &heights {
                for fmt in ALL_6_FORMATS {
                    // Use tight unpadded rows on odd heights, padded rows on even heights.
                    let pad = if h % 2 == 0 { 5 } else { 0 };
                    let rgb8 = make_rgb_from_rgba_fn(w, h, 8, fmt, pad, |_, _| {
                        [
                            rng.next_u16_bounded(255),
                            rng.next_u16_bounded(255),
                            rng.next_u16_bounded(255),
                            rng.next_u16_bounded(255),
                        ]
                    });
                    for pf in ALL_PFS {
                        let (mc, range) = match (w + h) % 6 {
                            0 => (MatrixCoefficients::Bt601, YuvRange::Full),
                            1 => (MatrixCoefficients::Bt709, YuvRange::Limited),
                            2 => (MatrixCoefficients::Bt2020Ncl, YuvRange::Full),
                            3 => (MatrixCoefficients::Fcc, YuvRange::Limited),
                            4 => (MatrixCoefficients::Identity, YuvRange::Full),
                            _ => (MatrixCoefficients::Ycgco, YuvRange::Limited),
                        };
                        verify_all_paths_0ulp(
                            &rgb8,
                            8,
                            pf,
                            range,
                            mc,
                            true,
                            &format!("8b {w}x{h} {fmt:?} {pf:?} {mc:?} {range:?}"),
                        );
                    }
                }
                // Also verify tight unpadded >8-bit (`10` and `16`-bit) 3ch + 4ch across ALL
                // widths x heights x subsamplings (specifically covers `Rgb`/`Bgr` u16 tail!).
                for (depth, fmt) in [(10u8, Format::Rgb), (12, Format::Bgra), (16, Format::Bgr)] {
                    let max_v = ((1u32 << depth) - 1) as u16;
                    let rgb16 = make_rgb_from_rgba_fn(w, h, depth, fmt, 0, |_, _| {
                        [
                            rng.next_u16_bounded(max_v),
                            rng.next_u16_bounded(max_v),
                            rng.next_u16_bounded(max_v),
                            rng.next_u16_bounded(max_v),
                        ]
                    });
                    for pf in ALL_PFS {
                        verify_all_paths_0ulp(
                            &rgb16,
                            depth,
                            pf,
                            YuvRange::Limited,
                            MatrixCoefficients::Bt709,
                            fmt.has_alpha(),
                            &format!("u16 d{depth} {w}x{h} {fmt:?} {pf:?}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rgb_to_yuv_avx2_all_depth_matrix_range_combinations_0ulp() {
        let depths = [8u8, 10, 12, 16];
        let mut rng = SplitMix64(0xA11C_E555_DEAD_BEEF);
        // Width 33 = 16 (unrolled) + 16 (unrolled) + 1 (scalar tail); height 3 tests paired + odd
        // row in 420; also test width 25 = 16 + 8 (8-lane step) + 1 (scalar tail)!
        for &w in &[25u32, 33] {
            let h = 3u32;
            for &in_d in &depths {
                let max_v = ((1u32 << in_d) - 1) as u16;
                let corners = [
                    0,
                    1,
                    127.min(max_v),
                    128.min(max_v),
                    254.min(max_v),
                    255.min(max_v),
                    max_v / 2,
                    max_v.saturating_sub(1),
                    max_v,
                ];
                for (fi, fmt) in ALL_6_FORMATS.into_iter().enumerate() {
                    let mut idx = 0usize;
                    let rgb = make_rgb_from_rgba_fn(w, h, in_d, fmt, 3, |_, _| {
                        idx += 1;
                        if idx <= corners.len() * 3 {
                            let c0 = corners[idx % corners.len()];
                            let c1 = corners[(idx / 3) % corners.len()];
                            let c2 = corners[(idx * 5) % corners.len()];
                            let ca = if in_d > 8 && idx % 7 == 0 {
                                65535 // Verify verbatim u16 alpha and YcgcoRe clamping on >max!
                            } else {
                                corners[(idx * 7) % corners.len()]
                            };
                            [c0, c1, c2, ca]
                        } else {
                            [
                                rng.next_u16_bounded(max_v),
                                rng.next_u16_bounded(max_v),
                                rng.next_u16_bounded(max_v),
                                rng.next_u16_bounded(max_v),
                            ]
                        }
                    });
                    for &out_d in &depths {
                        for range in [YuvRange::Full, YuvRange::Limited] {
                            // Cover every matrix across the format ring, and test ALL 7 matrices
                            // (+ YcgcoRo) exhaustively on Rgba and Rgb!
                            let mcs: &[MatrixCoefficients] =
                                if matches!(fmt, Format::Rgba | Format::Rgb) {
                                    &[
                                        MatrixCoefficients::Bt601,
                                        MatrixCoefficients::Bt709,
                                        MatrixCoefficients::Bt2020Ncl,
                                        MatrixCoefficients::Fcc,
                                        MatrixCoefficients::Identity,
                                        MatrixCoefficients::Ycgco,
                                        MatrixCoefficients::YcgcoRe,
                                        MatrixCoefficients::YcgcoRo,
                                    ]
                                } else {
                                    &ALL_MCS[fi % ALL_MCS.len()..=fi % ALL_MCS.len()]
                                };
                            for &mc in mcs {
                                if matches!(
                                    mc,
                                    MatrixCoefficients::YcgcoRe | MatrixCoefficients::YcgcoRo
                                ) && out_d == 8
                                {
                                    // YcgcoRe requires >8-bit YUV (`in_d + 2 == out_d` in AVIF).
                                    continue;
                                }
                                for pf in ALL_PFS {
                                    let ctx = format!(
                                        "{w}x{h} {fmt:?} {in_d}->{out_d} {range:?} {mc:?} {pf:?}"
                                    );
                                    verify_all_paths_0ulp(&rgb, out_d, pf, range, mc, true, &ctx);
                                }
                            }
                        }

                        // End-to-end `rgb.convert_to_yuv` check (uses `Bt709` + `Full` so native
                        // `rgb_impl::rgb_to_yuv` + `alpha_imported` always run rather than libyuv).
                        if matches!(fmt, Format::Rgba | Format::Abgr | Format::Rgb) && w == 25 {
                            for pf in ALL_PFS {
                                let mut e2e = alloc_sentinel_yuv(
                                    w,
                                    h,
                                    out_d,
                                    pf,
                                    YuvRange::Full,
                                    MatrixCoefficients::Bt709,
                                    true,
                                );
                                let mut ref_e2e = alloc_sentinel_yuv(
                                    w,
                                    h,
                                    out_d,
                                    pf,
                                    YuvRange::Full,
                                    MatrixCoefficients::Bt709,
                                    true,
                                );
                                rgb.convert_to_yuv(&mut e2e).unwrap();
                                ref_rgb_to_yuv_floor(&rgb, &mut ref_e2e, true);
                                assert_images_bit_exact(
                                    &e2e,
                                    &ref_e2e,
                                    &format!("convert_to_yuv {fmt:?} {pf:?} {in_d}->{out_d}"),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rgb_to_yuv_rgba1010102_fallback_and_pointer_buffers() {
        // 1. `Rgba1010102` must cleanly bypass AVX2 (`Ok(None)`), disallow fused alpha, and fail
        // upfront with `Err(AvifError::NotImplemented)` in `convert_to_yuv` without freeing or
        // overwriting pre-existing planes on `yuv`.
        let rgb_1010102 = rgb::Image {
            width: 16,
            height: 2,
            depth: 10,
            format: Format::Rgba1010102,
            pixels: Some(Pixels::Buffer16(vec![0xFFFFu16; 16 * 2 * 2])),
            row_bytes: 16 * 4,
            ..Default::default()
        };
        let mut yuv = alloc_sentinel_yuv(
            16,
            2,
            10,
            PixelFormat::Yuv444,
            YuvRange::Full,
            MatrixCoefficients::Bt709,
            true,
        );
        assert!(!can_fuse_alpha(&rgb_1010102, &yuv));
        #[cfg(target_arch = "x86_64")]
        assert_eq!(
            avx2::try_rgb_to_yuv_uninit_with_threads(&rgb_1010102, &mut yuv, true, 1).unwrap(),
            None
        );
        assert!(matches!(
            rgb_1010102.convert_to_yuv(&mut yuv),
            Err(AvifError::NotImplemented)
        ));
        assert_eq!(yuv.row16(Plane::Y, 0).unwrap()[0], 0xBEEF);

        // 2. Verify raw CAPI `Pixels::Pointer` / `Pixels::Pointer16` RGB inputs with tight unpadded
        // strides also match reference 0-ULP across 3ch (`Rgb`) and 4ch (`Bgra`).
        let mut raw8 = [0u8; 17 * 3 * 3];
        for (i, b) in raw8.iter_mut().enumerate() {
            *b = (i.wrapping_mul(37) ^ (i >> 2)) as u8;
        }
        let ptr_rgb8 = rgb::Image {
            width: 17,
            height: 3,
            depth: 8,
            format: Format::Rgb,
            pixels: Some(Pixels::from_raw_pointer(raw8.as_mut_ptr(), 8, 17 * 3, 3).unwrap()),
            row_bytes: 17 * 3,
            ..Default::default()
        };
        verify_all_paths_0ulp(
            &ptr_rgb8,
            8,
            PixelFormat::Yuv444,
            YuvRange::Full,
            MatrixCoefficients::Bt601,
            false,
            "pointer8_rgb",
        );

        let mut raw16 = [0u16; 19 * 2 * 4];
        for (i, w16) in raw16.iter_mut().enumerate() {
            *w16 = ((i.wrapping_mul(193)) & 0x03FF) as u16;
        }
        let ptr_bgra16 = rgb::Image {
            width: 19,
            height: 2,
            depth: 10,
            format: Format::Bgra,
            pixels: Some(
                Pixels::from_raw_pointer(raw16.as_mut_ptr().cast(), 10, 19 * 8, 2).unwrap(),
            ),
            row_bytes: 19 * 8,
            ..Default::default()
        };
        verify_all_paths_0ulp(
            &ptr_bgra16,
            10,
            PixelFormat::Yuv444,
            YuvRange::Limited,
            MatrixCoefficients::Bt2020Ncl,
            true,
            "pointer16_bgra",
        );
    }

    #[test]
    fn rgb_to_yuv_uninit_matches_reference_0ulp_and_pads_borders() {
        fn check_pad8(img: &image::Image, plane: Plane, expected_pad: u8) {
            let Some(Pixels::Buffer(buf)) = &img.planes[plane.as_usize()] else {
                panic!("expected 8-bit Pixels::Buffer on {plane:?}");
            };
            let w = img.width(plane);
            let h = img.height(plane);
            let stride = img.row_bytes[plane.as_usize()] as usize;
            let padded_h = crate::internal_utils::round2_usize(h);
            assert_eq!(buf.len(), stride * padded_h);
            for y in 0..h {
                for x in w..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
            for y in h..padded_h {
                for x in 0..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
        }

        fn check_pad16(img: &image::Image, plane: Plane, expected_pad: u16) {
            let Some(Pixels::Buffer16(buf)) = &img.planes[plane.as_usize()] else {
                panic!("expected 16-bit Pixels::Buffer16 on {plane:?}");
            };
            let w = img.width(plane);
            let h = img.height(plane);
            let stride = (img.row_bytes[plane.as_usize()] / 2) as usize;
            let padded_h = crate::internal_utils::round2_usize(h);
            assert_eq!(buf.len(), stride * padded_h);
            for y in 0..h {
                for x in w..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
            for y in h..padded_h {
                for x in 0..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
        }

        let pat_rgb = |w: u32, h: u32, depth: u8, format: Format| -> rgb::Image {
            let max_c = (1u32 << depth) - 1;
            make_rgb_from_rgba_fn(w, h, depth, format, 3, |x, y| {
                let b = x.wrapping_mul(53).wrapping_add(y.wrapping_mul(97));
                [
                    (b & max_c) as u16,
                    ((b.wrapping_mul(3) ^ 17) & max_c) as u16,
                    ((b.wrapping_mul(7) ^ 41) & max_c) as u16,
                    ((b.wrapping_mul(11) ^ 5) & max_c) as u16,
                ]
            })
        };

        for &(w, h) in &[(1u32, 1u32), (5, 3), (17, 3)] {
            for &pf in &[
                PixelFormat::Yuv444,
                PixelFormat::Yuv422,
                PixelFormat::Yuv420,
                PixelFormat::Yuv400,
            ] {
                let rgb8 = pat_rgb(w, h, 8, Format::Rgba);
                let mut ref8 = alloc_sentinel_yuv(
                    w,
                    h,
                    8,
                    pf,
                    YuvRange::Full,
                    MatrixCoefficients::Bt601,
                    true,
                );
                rgb8.convert_to_yuv(&mut ref8).unwrap();

                let mut got8 = image::Image {
                    width: w,
                    height: h,
                    depth: 8,
                    yuv_format: pf,
                    yuv_range: YuvRange::Full,
                    matrix_coefficients: MatrixCoefficients::Bt601,
                    ..Default::default()
                };
                rgb8.convert_to_yuv(&mut got8).unwrap();
                assert_images_bit_exact(&got8, &ref8, &format!("uninit8 {pf:?} {w}x{h}"));
                check_pad8(&got8, Plane::Y, 0);
                if pf != PixelFormat::Yuv400 {
                    check_pad8(&got8, Plane::U, 0);
                    check_pad8(&got8, Plane::V, 0);
                } else {
                    assert!(!got8.has_plane(Plane::U));
                    assert!(!got8.has_plane(Plane::V));
                }
                check_pad8(&got8, Plane::A, 255);

                let rgb10 = pat_rgb(w, h, 10, Format::Bgra);
                let mut ref10 = alloc_sentinel_yuv(
                    w,
                    h,
                    10,
                    pf,
                    YuvRange::Limited,
                    MatrixCoefficients::Bt709,
                    true,
                );
                rgb10.convert_to_yuv(&mut ref10).unwrap();

                let mut got10 = image::Image {
                    width: w,
                    height: h,
                    depth: 10,
                    yuv_format: pf,
                    yuv_range: YuvRange::Limited,
                    matrix_coefficients: MatrixCoefficients::Bt709,
                    ..Default::default()
                };
                rgb10.convert_to_yuv(&mut got10).unwrap();
                assert_images_bit_exact(&got10, &ref10, &format!("uninit10 {pf:?} {w}x{h}"));
                check_pad16(&got10, Plane::Y, 0);
                if pf != PixelFormat::Yuv400 {
                    check_pad16(&got10, Plane::U, 0);
                    check_pad16(&got10, Plane::V, 0);
                }
                check_pad16(&got10, Plane::A, 1023);
            }
        }

        // Verify cross-depth alpha (16-bit RGBA -> 8-bit YUV): YUV stays uninit-allocated via
        // AVX2 while non-fusible alpha falls back cleanly to `import_alpha_uninit`.
        let rgb16 = pat_rgb(17, 3, 16, Format::Rgba);
        let mut ref_cross = alloc_sentinel_yuv(
            17,
            3,
            8,
            PixelFormat::Yuv420,
            YuvRange::Full,
            MatrixCoefficients::Bt709,
            true,
        );
        rgb16.convert_to_yuv(&mut ref_cross).unwrap();
        let mut got_cross = image::Image {
            width: 17,
            height: 3,
            depth: 8,
            yuv_format: PixelFormat::Yuv420,
            yuv_range: YuvRange::Full,
            matrix_coefficients: MatrixCoefficients::Bt709,
            ..Default::default()
        };
        rgb16.convert_to_yuv(&mut got_cross).unwrap();
        assert_images_bit_exact(&got_cross, &ref_cross, "uninit cross 16->8");
        check_pad8(&got_cross, Plane::Y, 0);
        check_pad8(&got_cross, Plane::U, 0);
        check_pad8(&got_cross, Plane::V, 0);
        check_pad8(&got_cross, Plane::A, 255);
    }

    #[test]
    fn avx2_coalesced_and_multi_thread_0ulp() {
        fn check_pad8(img: &image::Image, plane: Plane, expected_pad: u8) {
            let Some(Pixels::Buffer(buf)) = &img.planes[plane.as_usize()] else {
                panic!("expected 8-bit Pixels::Buffer on {plane:?}");
            };
            let w = img.width(plane);
            let h = img.height(plane);
            let stride = img.row_bytes[plane.as_usize()] as usize;
            let padded_h = crate::internal_utils::round2_usize(h);
            assert_eq!(buf.len(), stride * padded_h);
            for y in 0..h {
                for x in w..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
            for y in h..padded_h {
                for x in 0..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
        }

        fn check_pad16(img: &image::Image, plane: Plane, expected_pad: u16) {
            let Some(Pixels::Buffer16(buf)) = &img.planes[plane.as_usize()] else {
                panic!("expected 16-bit Pixels::Buffer16 on {plane:?}");
            };
            let w = img.width(plane);
            let h = img.height(plane);
            let stride = (img.row_bytes[plane.as_usize()] / 2) as usize;
            let padded_h = crate::internal_utils::round2_usize(h);
            assert_eq!(buf.len(), stride * padded_h);
            for y in 0..h {
                for x in w..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
            for y in h..padded_h {
                for x in 0..stride {
                    assert_eq!(buf[y * stride + x], expected_pad);
                }
            }
        }

        let pat_rgb = |w: u32, h: u32, depth: u8, format: Format| -> rgb::Image {
            let max_c = (1u32 << depth) - 1;
            make_rgb_from_rgba_fn(w, h, depth, format, 3, |x, y| {
                let b = x.wrapping_mul(53).wrapping_add(y.wrapping_mul(97));
                [
                    (b & max_c) as u16,
                    ((b.wrapping_mul(3) ^ 17) & max_c) as u16,
                    ((b.wrapping_mul(7) ^ 41) & max_c) as u16,
                    ((b.wrapping_mul(11) ^ 5) & max_c) as u16,
                ]
            })
        };

        // Check coalesced tightly-packed rows against padded-stride non-coalesced rows (0-ULP).
        for &(width, height) in &[(3u32, 11u32), (17, 5), (32, 4), (33, 7)] {
            for &depth in &[8u8, 10] {
                for &yuv_format in &[PixelFormat::Yuv444, PixelFormat::Yuv400] {
                    for &rgb_format in &[Format::Rgba, Format::Bgra, Format::Rgb, Format::Bgr] {
                        let rgb_packed = pat_rgb(width, height, depth, rgb_format);
                        let bpp = rgb_packed.pixel_size();
                        let padded_stride = (width * bpp) + 16;
                        let mut rgb_padded = rgb::Image {
                            width,
                            height,
                            depth,
                            format: rgb_format,
                            row_bytes: padded_stride,
                            ..Default::default()
                        };
                        if depth == 8 {
                            let mut buf = vec![0xA5u8; (padded_stride * height) as usize];
                            for row in 0..height {
                                let src_row = rgb_packed.row(row).unwrap();
                                let dst_off = (row * padded_stride) as usize;
                                buf[dst_off..dst_off + src_row.len()].copy_from_slice(src_row);
                            }
                            rgb_padded.pixels = Some(Pixels::Buffer(buf));
                        } else {
                            let elem_stride = (padded_stride / 2) as usize;
                            let mut buf = vec![0xA5u16; elem_stride * height as usize];
                            for row in 0..height {
                                let src_row = rgb_packed.row16(row).unwrap();
                                let dst_off = row as usize * elem_stride;
                                buf[dst_off..dst_off + src_row.len()].copy_from_slice(src_row);
                            }
                            rgb_padded.pixels = Some(Pixels::Buffer16(buf));
                        }

                        for &yuv_range in &[YuvRange::Full, YuvRange::Limited] {
                            for &matrix in &[
                                MatrixCoefficients::Bt601,
                                MatrixCoefficients::Bt709,
                                MatrixCoefficients::Identity,
                                MatrixCoefficients::Ycgco,
                                MatrixCoefficients::YcgcoRe,
                            ] {
                                if yuv_format == PixelFormat::Yuv400
                                    && matches!(
                                        matrix,
                                        MatrixCoefficients::Identity | MatrixCoefficients::YcgcoRe
                                    )
                                {
                                    continue;
                                }
                                if matrix == MatrixCoefficients::YcgcoRe && depth != 10 {
                                    continue;
                                }
                                let alloc_alpha = rgb_format.has_alpha();
                                let mut got_coalesced = image::Image {
                                    width,
                                    height,
                                    depth,
                                    yuv_format,
                                    yuv_range,
                                    matrix_coefficients: matrix,
                                    ..Default::default()
                                };
                                let mut got_strided = image::Image {
                                    width,
                                    height,
                                    depth,
                                    yuv_format,
                                    yuv_range,
                                    matrix_coefficients: matrix,
                                    ..Default::default()
                                };
                                #[cfg(target_arch = "x86_64")]
                                {
                                    if std::arch::is_x86_feature_detected!("avx2") {
                                        let ok_c = avx2::try_rgb_to_yuv_uninit_with_threads(
                                            &rgb_packed,
                                            &mut got_coalesced,
                                            alloc_alpha,
                                            1,
                                        )
                                        .unwrap();
                                        let ok_s = avx2::try_rgb_to_yuv_uninit_with_threads(
                                            &rgb_padded,
                                            &mut got_strided,
                                            alloc_alpha,
                                            1,
                                        )
                                        .unwrap();
                                        assert_eq!(ok_c, Some(alloc_alpha));
                                        assert_eq!(ok_s, Some(alloc_alpha));
                                    } else {
                                        let ok_c = rgb_to_yuv_uninit_with_threads(
                                            &rgb_packed,
                                            &mut got_coalesced,
                                            alloc_alpha,
                                            1,
                                        )
                                        .unwrap();
                                        let ok_s = rgb_to_yuv_uninit_with_threads(
                                            &rgb_padded,
                                            &mut got_strided,
                                            alloc_alpha,
                                            1,
                                        )
                                        .unwrap();
                                        assert_eq!(ok_c, alloc_alpha);
                                        assert_eq!(ok_s, alloc_alpha);
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    let ok_c = rgb_to_yuv_uninit_with_threads(
                                        &rgb_packed,
                                        &mut got_coalesced,
                                        alloc_alpha,
                                        1,
                                    )
                                    .unwrap();
                                    let ok_s = rgb_to_yuv_uninit_with_threads(
                                        &rgb_padded,
                                        &mut got_strided,
                                        alloc_alpha,
                                        1,
                                    )
                                    .unwrap();
                                    assert_eq!(ok_c, alloc_alpha);
                                    assert_eq!(ok_s, alloc_alpha);
                                }
                                assert_images_bit_exact(
                                    &got_coalesced,
                                    &got_strided,
                                    &format!(
                                        "coalesced {width}x{height} d={depth} {yuv_format:?} \
                                         {rgb_format:?} {yuv_range:?} {matrix:?}"
                                    ),
                                );
                                if depth == 8 {
                                    check_pad8(&got_coalesced, Plane::Y, 0);
                                    if yuv_format != PixelFormat::Yuv400 {
                                        check_pad8(&got_coalesced, Plane::U, 0);
                                        check_pad8(&got_coalesced, Plane::V, 0);
                                    }
                                    if alloc_alpha {
                                        check_pad8(&got_coalesced, Plane::A, 255);
                                    }
                                } else {
                                    check_pad16(&got_coalesced, Plane::Y, 0);
                                    if yuv_format != PixelFormat::Yuv400 {
                                        check_pad16(&got_coalesced, Plane::U, 0);
                                        check_pad16(&got_coalesced, Plane::V, 0);
                                    }
                                    if alloc_alpha {
                                        check_pad16(&got_coalesced, Plane::A, (1u16 << depth) - 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check multi-threaded stripe execution (>= 512 * 512) with odd dimensions.
        for &(width, height) in &[(641u32, 521u32), (513, 515)] {
            for &(depth, rgb_format, yuv_format, yuv_range, matrix) in &[
                (
                    8u8,
                    Format::Rgba,
                    PixelFormat::Yuv444,
                    YuvRange::Full,
                    MatrixCoefficients::Bt601,
                ),
                (
                    8,
                    Format::Bgra,
                    PixelFormat::Yuv422,
                    YuvRange::Limited,
                    MatrixCoefficients::Bt709,
                ),
                (
                    8,
                    Format::Argb,
                    PixelFormat::Yuv420,
                    YuvRange::Full,
                    MatrixCoefficients::Bt601,
                ),
                (
                    8,
                    Format::Abgr,
                    PixelFormat::Yuv420,
                    YuvRange::Limited,
                    MatrixCoefficients::Bt709,
                ),
                (
                    8,
                    Format::Bgr,
                    PixelFormat::Yuv444,
                    YuvRange::Full,
                    MatrixCoefficients::Identity,
                ),
                (
                    8,
                    Format::Rgb,
                    PixelFormat::Yuv400,
                    YuvRange::Full,
                    MatrixCoefficients::Ycgco,
                ),
                (
                    10,
                    Format::Rgba,
                    PixelFormat::Yuv420,
                    YuvRange::Full,
                    MatrixCoefficients::YcgcoRe,
                ),
                (
                    10,
                    Format::Bgr,
                    PixelFormat::Yuv422,
                    YuvRange::Full,
                    MatrixCoefficients::Bt709,
                ),
            ] {
                let alloc_alpha = rgb_format.has_alpha();
                let rgb = pat_rgb(width, height, depth, rgb_format);
                let mut ref_st = image::Image {
                    width,
                    height,
                    depth,
                    yuv_format,
                    yuv_range,
                    matrix_coefficients: matrix,
                    ..Default::default()
                };
                let mut got_mt = image::Image {
                    width,
                    height,
                    depth,
                    yuv_format,
                    yuv_range,
                    matrix_coefficients: matrix,
                    ..Default::default()
                };
                #[cfg(target_arch = "x86_64")]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        let st_ok = avx2::try_rgb_to_yuv_uninit_with_threads(
                            &rgb,
                            &mut ref_st,
                            alloc_alpha,
                            1,
                        )
                        .unwrap();
                        let mt_ok = avx2::try_rgb_to_yuv_uninit_with_threads(
                            &rgb,
                            &mut got_mt,
                            alloc_alpha,
                            4,
                        )
                        .unwrap();
                        assert_eq!(st_ok, Some(alloc_alpha));
                        assert_eq!(mt_ok, Some(alloc_alpha));
                    } else {
                        let st_ok =
                            rgb_to_yuv_uninit_with_threads(&rgb, &mut ref_st, alloc_alpha, 1)
                                .unwrap();
                        let mt_ok =
                            rgb_to_yuv_uninit_with_threads(&rgb, &mut got_mt, alloc_alpha, 4)
                                .unwrap();
                        assert_eq!(st_ok, alloc_alpha);
                        assert_eq!(mt_ok, alloc_alpha);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let st_ok =
                        rgb_to_yuv_uninit_with_threads(&rgb, &mut ref_st, alloc_alpha, 1).unwrap();
                    let mt_ok =
                        rgb_to_yuv_uninit_with_threads(&rgb, &mut got_mt, alloc_alpha, 4).unwrap();
                    assert_eq!(st_ok, alloc_alpha);
                    assert_eq!(mt_ok, alloc_alpha);
                }
                assert_images_bit_exact(
                    &got_mt,
                    &ref_st,
                    &format!(
                        "mt {width}x{height} d={depth} {rgb_format:?} {yuv_format:?} \
                         {yuv_range:?} {matrix:?}"
                    ),
                );
                if depth == 8 {
                    check_pad8(&got_mt, Plane::Y, 0);
                    if yuv_format != PixelFormat::Yuv400 {
                        check_pad8(&got_mt, Plane::U, 0);
                        check_pad8(&got_mt, Plane::V, 0);
                    }
                    if alloc_alpha {
                        check_pad8(&got_mt, Plane::A, 255);
                    }
                } else {
                    check_pad16(&got_mt, Plane::Y, 0);
                    if yuv_format != PixelFormat::Yuv400 {
                        check_pad16(&got_mt, Plane::U, 0);
                        check_pad16(&got_mt, Plane::V, 0);
                    }
                    if alloc_alpha {
                        check_pad16(&got_mt, Plane::A, (1u16 << depth) - 1);
                    }
                }
            }
        }
    }

    #[test]
    fn cross_depth_alpha_0_to_65535_bit_exact() {
        let depths = [8u8, 10, 12, 16];
        let pfs = [
            PixelFormat::Yuv444,
            PixelFormat::Yuv422,
            PixelFormat::Yuv420,
            PixelFormat::Yuv400,
        ];
        // Exhaustively test every alpha value 0..=65535 (or 0..=255 for 8-bit) across a multi-row
        // odd-width image (width=1027 -> covers 16px, 8px, and 3px tail across rows) against
        // `Image::import_alpha_from`.
        let w = 1027u32;
        let h = 64u32;
        let formats = [Format::Rgba, Format::Bgra, Format::Argb, Format::Abgr];
        for (idx, &rgb_d) in depths.iter().enumerate() {
            for (jdx, &yuv_d) in depths.iter().enumerate() {
                if rgb_d == yuv_d {
                    continue;
                }
                let fmt = formats[(idx + jdx) % formats.len()];
                let pf = pfs[(idx * 3 + jdx) % pfs.len()];
                let rgb = make_rgb_from_rgba_fn(w, h, rgb_d, fmt, 0, |x, y| {
                    let v = ((y * w + x) & 0xFFFF) as u16;
                    let a = if rgb_d == 8 { v & 0xFF } else { v };
                    let c_max = ((1u32 << rgb_d) - 1) as u16;
                    [
                        v.wrapping_mul(3) & c_max,
                        v.wrapping_mul(5) & c_max,
                        v.wrapping_mul(7) & c_max,
                        a,
                    ]
                });
                let mut ref_img = image::Image {
                    width: w,
                    height: h,
                    depth: yuv_d,
                    yuv_format: pf,
                    yuv_range: if (idx + jdx) % 2 == 0 {
                        YuvRange::Full
                    } else {
                        YuvRange::Limited
                    },
                    matrix_coefficients: MatrixCoefficients::Bt709,
                    ..Default::default()
                };
                ref_img.allocate_planes(Category::Alpha).unwrap();
                ref_img.import_alpha_from(&rgb).unwrap();

                let mut got_img = image::Image {
                    width: w,
                    height: h,
                    depth: yuv_d,
                    yuv_format: pf,
                    yuv_range: ref_img.yuv_range,
                    matrix_coefficients: MatrixCoefficients::Bt709,
                    ..Default::default()
                };
                let fused = rgb_to_yuv_uninit_with_threads(&rgb, &mut got_img, true, 4).unwrap();
                assert!(
                    fused,
                    "expected cross-depth fusion for {rgb_d}->{yuv_d} {pf:?}"
                );

                for y in 0..h {
                    if yuv_d > 8 {
                        assert_eq!(
                            got_img.row16(Plane::A, y).unwrap(),
                            ref_img.row16(Plane::A, y).unwrap(),
                            "alpha u16 mismatch {rgb_d}->{yuv_d} {fmt:?} {pf:?} row {y}"
                        );
                    } else {
                        assert_eq!(
                            got_img.row(Plane::A, y).unwrap(),
                            ref_img.row(Plane::A, y).unwrap(),
                            "alpha u8 mismatch {rgb_d}->{yuv_d} {fmt:?} {pf:?} row {y}"
                        );
                    }
                }

                // Also verify scalar Yuv444 cross-depth fused alpha on odd width with padding.
                let mut got_scalar_444 = alloc_sentinel_yuv(
                    27,
                    5,
                    yuv_d,
                    PixelFormat::Yuv444,
                    YuvRange::Full,
                    MatrixCoefficients::Bt709,
                    true,
                );
                let rgb_small = make_rgb_from_rgba_fn(27, 5, rgb_d, fmt, 3, |x, y| {
                    let v = ((y * 27 + x).wrapping_mul(521) & 0xFFFF) as u16;
                    let c_max = ((1u32 << rgb_d) - 1) as u16;
                    [
                        v & c_max,
                        v.wrapping_mul(3) & c_max,
                        v.wrapping_mul(7) & c_max,
                        if rgb_d == 8 { v & 0xFF } else { v },
                    ]
                });
                let mut ref_small = alloc_sentinel_yuv(
                    27,
                    5,
                    yuv_d,
                    PixelFormat::Yuv444,
                    YuvRange::Full,
                    MatrixCoefficients::Bt709,
                    true,
                );
                ref_small.import_alpha_from(&rgb_small).unwrap();
                assert!(rgb_to_yuv_scalar(&rgb_small, &mut got_scalar_444).unwrap());
                for y in 0..5 {
                    if yuv_d > 8 {
                        assert_eq!(
                            got_scalar_444.row16(Plane::A, y).unwrap(),
                            ref_small.row16(Plane::A, y).unwrap(),
                        );
                    } else {
                        assert_eq!(
                            got_scalar_444.row(Plane::A, y).unwrap(),
                            ref_small.row(Plane::A, y).unwrap(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rgb_gray_to_yuv_avx2_0ulp() {
        fn ref_gray_scalar(rgb: &rgb::Image, image: &mut image::Image) {
            let rgb_channel_count = rgb.channel_count() as usize;
            let gray_offset = rgb.format.r_offset();
            let rgb_max_channel_f = rgb.max_channel_f();
            let (bias_y, range_y) = bias_and_range_y(image);
            let yuv_max_channel = image.max_channel();
            for j in 0..image.height {
                for i in 0..image.width as usize {
                    let gray_pixel = if rgb.depth == 8 {
                        let src = rgb.row(j).unwrap();
                        src[(i * rgb_channel_count) + gray_offset] as f32 / rgb_max_channel_f
                    } else {
                        let src = rgb.row16(j).unwrap();
                        src[(i * rgb_channel_count) + gray_offset] as f32 / rgb_max_channel_f
                    };
                    let y_val = to_unorm(bias_y, range_y, yuv_max_channel, gray_pixel);
                    if image.depth == 8 {
                        image.row_mut(Plane::Y, j).unwrap()[i] = y_val as u8;
                    } else {
                        image.row16_mut(Plane::Y, j).unwrap()[i] = y_val;
                    }
                }
            }
            let chroma_value = (image.max_channel() / 2) + 1;
            image.fill_plane_with_value(Plane::U, chroma_value).unwrap();
            image.fill_plane_with_value(Plane::V, chroma_value).unwrap();
        }

        let make_gray_rgb = |w: u32,
                             h: u32,
                             depth: u8,
                             fmt: Format,
                             extra_pad_elems: usize,
                             mut fill: Box<dyn FnMut(u32, u32) -> (u16, u16)>|
         -> rgb::Image {
            let ch = fmt.channel_count() as usize;
            let row_elems = (w as usize) * ch + extra_pad_elems;
            if depth == 8 {
                let mut buf = vec![0xAAu8; row_elems * (h as usize)];
                for y in 0..h {
                    for x in 0..w {
                        let (g, a) = fill(x, y);
                        let idx = (y as usize) * row_elems + (x as usize) * ch;
                        match fmt {
                            Format::Gray => buf[idx] = g as u8,
                            Format::GrayA => {
                                buf[idx] = g as u8;
                                buf[idx + 1] = a as u8;
                            }
                            Format::AGray => {
                                buf[idx] = a as u8;
                                buf[idx + 1] = g as u8;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                rgb::Image {
                    width: w,
                    height: h,
                    depth,
                    format: fmt,
                    pixels: Some(Pixels::Buffer(buf)),
                    row_bytes: row_elems as u32,
                    ..Default::default()
                }
            } else {
                let mut buf = vec![0xBEEFu16; row_elems * (h as usize)];
                for y in 0..h {
                    for x in 0..w {
                        let (g, a) = fill(x, y);
                        let idx = (y as usize) * row_elems + (x as usize) * ch;
                        match fmt {
                            Format::Gray => buf[idx] = g,
                            Format::GrayA => {
                                buf[idx] = g;
                                buf[idx + 1] = a;
                            }
                            Format::AGray => {
                                buf[idx] = a;
                                buf[idx + 1] = g;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                rgb::Image {
                    width: w,
                    height: h,
                    depth,
                    format: fmt,
                    pixels: Some(Pixels::Buffer16(buf)),
                    row_bytes: (row_elems * 2) as u32,
                    ..Default::default()
                }
            }
        };

        let fmts = [Format::Gray, Format::GrayA, Format::AGray];
        let depths = [8u8, 10, 12, 16];
        for &fmt in &fmts {
            for &rgb_d in &depths {
                for &yuv_d in &depths {
                    for &range in &[YuvRange::Full, YuvRange::Limited] {
                        for &(w, h, pad, pf) in &[
                            (1u32, 1u32, 0usize, PixelFormat::Yuv400),
                            (17u32, 3u32, 0usize, PixelFormat::Yuv400),
                            (33u32, 5u32, 3usize, PixelFormat::Yuv420),
                            (257u32, 4u32, 0usize, PixelFormat::Yuv444),
                        ] {
                            let rgb = make_gray_rgb(
                                w,
                                h,
                                rgb_d,
                                fmt,
                                pad,
                                Box::new(move |x, y| {
                                    let seq = (y * w + x) as u16;
                                    let g = if rgb_d == 8 {
                                        seq & 0xFF
                                    } else {
                                        // Mix in-range values plus occasional max-clamp u16=65535.
                                        if seq % 37 == 0 {
                                            65535
                                        } else {
                                            seq.wrapping_mul(257)
                                        }
                                    };
                                    (g, seq ^ 0x55)
                                }),
                            );
                            let mut ref_img = alloc_sentinel_yuv(
                                w,
                                h,
                                yuv_d,
                                pf,
                                range,
                                MatrixCoefficients::Bt601,
                                false,
                            );
                            let mut got_img = alloc_sentinel_yuv(
                                w,
                                h,
                                yuv_d,
                                pf,
                                range,
                                MatrixCoefficients::Bt601,
                                false,
                            );
                            ref_gray_scalar(&rgb, &mut ref_img);
                            rgb_gray_to_yuv(&rgb, &mut got_img).unwrap();
                            assert_images_bit_exact(
                                &got_img,
                                &ref_img,
                                &format!("gray {fmt:?} {rgb_d}->{yuv_d} {range:?} {w}x{h} {pf:?}"),
                            );
                        }
                    }
                }
            }
        }

        // Exhaustive 0..=65535 sample check across 16b -> 8b/10b/12b/16b.
        let rgb16 = make_gray_rgb(
            1029,
            64,
            16,
            Format::Gray,
            0,
            Box::new(|x, y| (((y * 1029 + x) & 0xFFFF) as u16, 0)),
        );
        for &yuv_d in &[8u8, 10, 12, 16] {
            for &range in &[YuvRange::Full, YuvRange::Limited] {
                let mut ref_img = alloc_sentinel_yuv(
                    1029,
                    64,
                    yuv_d,
                    PixelFormat::Yuv400,
                    range,
                    MatrixCoefficients::Bt709,
                    false,
                );
                let mut got_img = alloc_sentinel_yuv(
                    1029,
                    64,
                    yuv_d,
                    PixelFormat::Yuv400,
                    range,
                    MatrixCoefficients::Bt709,
                    false,
                );
                ref_gray_scalar(&rgb16, &mut ref_img);
                rgb_gray_to_yuv(&rgb16, &mut got_img).unwrap();
                assert_images_bit_exact(
                    &got_img,
                    &ref_img,
                    &format!("gray exhaustive 16->{yuv_d} {range:?}"),
                );
            }
        }
    }
}
