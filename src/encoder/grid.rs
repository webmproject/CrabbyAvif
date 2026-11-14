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

// Helpers to split a single image into the cells of a grid image.

use crate::image::*;
use crate::internal_utils::*;
use crate::utils::clap::CropRect;
use crate::*;

// Maximum number of columns or rows of an ImageGrid box (rows_minus_one and columns_minus_one
// are 8-bit fields).
const MAX_GRID_DIMENSION: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GridLayout {
    pub columns: u32,
    pub rows: u32,
    // Dimensions of all cells, except for the right-most column and the bottom-most row which may
    // be narrower and shorter respectively.
    pub cell_width: u32,
    pub cell_height: u32,
}

impl GridLayout {
    pub(crate) fn cell_count(&self) -> AvifResult<usize> {
        usize_from_u32(checked_mul!(self.columns, self.rows)?)
    }
}

// Returns the number of cells and the effective cell size along an axis of num_pixels samples
// split into cells of cell_size samples. A cell_size larger than num_pixels is clamped.
fn axis_layout(num_pixels: u32, cell_size: u32) -> AvifResult<(u32, u32)> {
    if num_pixels == 0 || cell_size == 0 {
        return AvifError::invalid_argument();
    }
    let cell_size = cell_size.min(num_pixels);
    let count = num_pixels.div_ceil(cell_size);
    if count > MAX_GRID_DIMENSION {
        return AvifError::invalid_image_grid(format!(
            "splitting {num_pixels} samples into cells of {cell_size} samples requires {count} \
             cells, which is more than the maximum of {MAX_GRID_DIMENSION}"
        ));
    }
    Ok((count, cell_size))
}

// Returns the layout of the grid obtained by splitting an image of image_width x image_height
// samples into cells of cell_width x cell_height samples. The cell dimensions are clamped to the
// image dimensions. MIAF constraints on the cell dimensions are not checked here.
pub(crate) fn compute_layout(
    image_width: u32,
    image_height: u32,
    cell_width: u32,
    cell_height: u32,
) -> AvifResult<GridLayout> {
    let (columns, cell_width) = axis_layout(image_width, cell_width)?;
    let (rows, cell_height) = axis_layout(image_height, cell_height)?;
    Ok(GridLayout {
        columns,
        rows,
        cell_width,
        cell_height,
    })
}

// Returns the cell size along an axis of gainmap_pixels samples of a gain map, given that the
// base image has base_pixels samples along that axis split into count cells of base_cell samples.
// The cell size is proportional to the gain map dimension, rounded up to an even number if the
// gain map is subsampled along that axis. If that does not lead to the same number of cells as
// the base image, ceil(gainmap_pixels / count) is used instead (rounded up to an even number if
// subsampled).
pub(crate) fn scale_cell_size(
    base_cell: u32,
    base_pixels: u32,
    gainmap_pixels: u32,
    count: u32,
    is_subsampled: bool,
) -> AvifResult<u32> {
    if base_cell == 0 || base_pixels == 0 || gainmap_pixels == 0 || count == 0 {
        return AvifError::invalid_argument();
    }
    let gainmap_pixels_u64 = u64::from(gainmap_pixels);
    let round_up_to_even = |size: u64| if is_subsampled && size % 2 == 1 { size + 1 } else { size };
    let cell_count = |size: u64| gainmap_pixels_u64.div_ceil(size.min(gainmap_pixels_u64));

    let proportional = round_up_to_even(
        (u64::from(base_cell) * gainmap_pixels_u64).div_ceil(u64::from(base_pixels)),
    );
    if cell_count(proportional) == u64::from(count) {
        return u32_from_u64(proportional.min(gainmap_pixels_u64));
    }
    let fallback = round_up_to_even(gainmap_pixels_u64.div_ceil(u64::from(count)));
    if cell_count(fallback) == u64::from(count) {
        return u32_from_u64(fallback.min(gainmap_pixels_u64));
    }
    AvifError::invalid_image_grid(format!(
        "the gain map dimension {gainmap_pixels} cannot be split into {count} cells"
    ))
}

// Returns layout.columns * layout.rows views of image in raster order. The views point to the
// pixels of image (no pixel data is copied). Exif, XMP and ICC data are not copied.
//
// The returned images must not outlive image and the pixel buffers of image must not be
// reallocated or freed while the views are alive.
pub(crate) fn split_into_views(image: &Image, layout: &GridLayout) -> AvifResult<Vec<Image>> {
    // The cell offsets must be aligned to the chroma subsampling. Report it as an invalid grid
    // (as MIAF requires even cell dimensions in that case) rather than as an invalid argument.
    if (layout.columns > 1
        && image.yuv_format.chroma_shift_x().0 != 0
        && layout.cell_width % 2 != 0)
        || (layout.rows > 1
            && image.yuv_format.chroma_shift_y() != 0
            && layout.cell_height % 2 != 0)
    {
        return AvifError::invalid_image_grid(format!(
            "grid cell width ({}) or height ({}) shall be even if chroma is subsampled in that \
             dimension. See MIAF (ISO/IEC 23000-22:2019), Section 7.3.11.4.2",
            layout.cell_width, layout.cell_height
        ));
    }
    let mut views: Vec<Image> = create_vec_exact(layout.cell_count()?)?;
    for row in 0..layout.rows {
        let y = checked_mul!(row, layout.cell_height)?;
        let height = layout.cell_height.min(checked_sub!(image.height, y)?);
        for column in 0..layout.columns {
            let x = checked_mul!(column, layout.cell_width)?;
            let width = layout.cell_width.min(checked_sub!(image.width, x)?);
            views.try_push(image.view(&CropRect {
                x,
                y,
                width,
                height,
            })?)?;
        }
    }
    Ok(views)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case(1000, 700, 512, 512, 2, 2, 512, 512 ; "remainder")]
    #[test_case(4000, 3000, 2000, 1500, 2, 2, 2000, 1500 ; "exact multiple")]
    #[test_case(300, 200, 512, 512, 1, 1, 300, 200 ; "clamped single cell")]
    #[test_case(2000, 300, 512, 512, 4, 1, 512, 300 ; "clamped height")]
    #[test_case(1000, 700, 32, 512, 32, 2, 32, 512 ; "small cells")]
    #[test_case(256, 256, 1, 1, 256, 256, 1, 1 ; "max grid dimensions")]
    #[test_case(u32::MAX, u32::MAX, u32::MAX, u32::MAX, 1, 1, u32::MAX, u32::MAX ; "max u32")]
    #[test_case(u32::MAX, 1, 1 << 24, 1, 256, 1, 1 << 24, 1 ; "max u32 grid")]
    #[allow(clippy::too_many_arguments)]
    fn compute_layout_ok(
        image_width: u32,
        image_height: u32,
        cell_width: u32,
        cell_height: u32,
        columns: u32,
        rows: u32,
        effective_cell_width: u32,
        effective_cell_height: u32,
    ) -> AvifResult<()> {
        assert_eq!(
            compute_layout(image_width, image_height, cell_width, cell_height)?,
            GridLayout {
                columns,
                rows,
                cell_width: effective_cell_width,
                cell_height: effective_cell_height,
            }
        );
        Ok(())
    }

    #[test_case(0, 100, 64, 64 ; "zero image width")]
    #[test_case(100, 0, 64, 64 ; "zero image height")]
    #[test_case(100, 100, 0, 64 ; "zero cell width")]
    #[test_case(100, 100, 64, 0 ; "zero cell height")]
    #[test_case(257, 100, 1, 100 ; "too many columns")]
    #[test_case(100, 70000, 100, 256 ; "too many rows")]
    fn compute_layout_error(
        image_width: u32,
        image_height: u32,
        cell_width: u32,
        cell_height: u32,
    ) {
        assert!(compute_layout(image_width, image_height, cell_width, cell_height).is_err());
    }

    #[test_case(1024, 4000, 1000, 4, false, 256 ; "proportional quarter")]
    #[test_case(1024, 3000, 750, 3, false, 256 ; "proportional quarter with remainder")]
    #[test_case(512, 1000, 500, 2, false, 256 ; "proportional half")]
    #[test_case(333, 999, 500, 3, false, 167 ; "proportional rounded up")]
    #[test_case(333, 999, 500, 3, true, 168 ; "proportional rounded up to even")]
    #[test_case(990, 1000, 10, 2, false, 5 ; "fallback")]
    #[test_case(990, 1000, 10, 2, true, 6 ; "fallback rounded up to even")]
    #[test_case(3000, 3000, 750, 1, false, 750 ; "single cell")]
    fn scale_cell_size_ok(
        base_cell: u32,
        base_pixels: u32,
        gainmap_pixels: u32,
        count: u32,
        is_subsampled: bool,
        expected: u32,
    ) -> AvifResult<()> {
        let cell = scale_cell_size(base_cell, base_pixels, gainmap_pixels, count, is_subsampled)?;
        assert_eq!(cell, expected);
        assert_eq!(gainmap_pixels.div_ceil(cell), count);
        Ok(())
    }

    #[test_case(0, 100, 50, 1, false ; "zero base cell")]
    #[test_case(100, 0, 50, 1, false ; "zero base pixels")]
    #[test_case(100, 100, 0, 1, false ; "zero gainmap pixels")]
    #[test_case(100, 100, 50, 0, false ; "zero count")]
    #[test_case(64, 256, 3, 4, false ; "fewer gainmap pixels than cells")]
    #[test_case(64, 256, 6, 4, true ; "fallback with subsampling")]
    fn scale_cell_size_error(
        base_cell: u32,
        base_pixels: u32,
        gainmap_pixels: u32,
        count: u32,
        is_subsampled: bool,
    ) {
        assert!(
            scale_cell_size(base_cell, base_pixels, gainmap_pixels, count, is_subsampled).is_err()
        );
    }

    fn create_image(width: u32, height: u32, yuv_format: PixelFormat) -> AvifResult<Image> {
        let mut image = Image {
            width,
            height,
            depth: 8,
            yuv_format,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        image.allocate_planes(Category::Alpha)?;
        image.alpha_present = true;
        for plane in ALL_PLANES {
            if !image.has_plane(plane) {
                continue;
            }
            let plane_data = image.plane_data(plane).ok_or(AvifError::NoContent)?;
            for y in 0..plane_data.height {
                for (x, pixel) in image.row_mut(plane, y)?.iter_mut().enumerate() {
                    *pixel = (x as u32 + y * 3 + plane as u32 * 7) as u8;
                }
            }
        }
        Ok(image)
    }

    #[test_case(PixelFormat::Yuv400)]
    #[test_case(PixelFormat::Yuv420)]
    #[test_case(PixelFormat::Yuv422)]
    #[test_case(PixelFormat::Yuv444)]
    fn split_into_views_geometry(yuv_format: PixelFormat) -> AvifResult<()> {
        let mut image = create_image(200, 150, yuv_format)?;
        image.icc.try_extend_from_slice(&[1, 2, 3])?;
        let layout = compute_layout(image.width, image.height, 80, 64)?;
        assert_eq!((layout.columns, layout.rows), (3, 3));
        let views = split_into_views(&image, &layout)?;
        assert_eq!(views.len(), 9);
        for (index, view) in views.iter().enumerate() {
            let column = index as u32 % layout.columns;
            let row = index as u32 / layout.columns;
            let (x, y) = (column * 80, row * 64);
            assert_eq!(view.width, if column == 2 { 40 } else { 80 });
            assert_eq!(view.height, if row == 2 { 22 } else { 64 });
            assert!(view.icc.is_empty());
            assert!(view.has_same_cicp(&image));
            for plane in ALL_PLANES {
                assert_eq!(view.has_plane(plane), image.has_plane(plane));
                if !view.has_plane(plane) {
                    continue;
                }
                let (plane_x, plane_y) = if plane == Plane::Y || plane == Plane::A {
                    (x as usize, y)
                } else {
                    (
                        yuv_format.apply_chroma_shift_x(x) as usize,
                        yuv_format.apply_chroma_shift_y(y),
                    )
                };
                for view_y in 0..view.height(plane) as u32 {
                    let view_row = view.row(plane, view_y)?;
                    let image_row = image.row(plane, plane_y + view_y)?;
                    assert_eq!(view_row, &image_row[plane_x..plane_x + view.width(plane)]);
                }
            }
        }
        Ok(())
    }

    #[test_case(PixelFormat::Yuv420, 200, 150, 81, 64, false ; "420 odd width")]
    #[test_case(PixelFormat::Yuv420, 200, 150, 80, 65, false ; "420 odd height")]
    #[test_case(PixelFormat::Yuv422, 200, 150, 81, 64, false ; "422 odd width")]
    #[test_case(PixelFormat::Yuv422, 200, 150, 80, 65, true ; "422 odd height")]
    #[test_case(PixelFormat::Yuv444, 200, 150, 81, 65, true ; "444 odd")]
    #[test_case(PixelFormat::Yuv400, 200, 150, 81, 65, true ; "400 odd")]
    #[test_case(PixelFormat::Yuv420, 81, 65, 81, 65, true ; "420 odd single cell")]
    #[test_case(PixelFormat::Yuv420, 81, 150, 81, 64, true ; "420 odd width single column")]
    fn split_into_views_odd_cells(
        yuv_format: PixelFormat,
        width: u32,
        height: u32,
        cell_width: u32,
        cell_height: u32,
        expect_success: bool,
    ) -> AvifResult<()> {
        let image = create_image(width, height, yuv_format)?;
        let layout = compute_layout(image.width, image.height, cell_width, cell_height)?;
        let res = split_into_views(&image, &layout);
        assert_eq!(res.is_ok(), expect_success);
        if let Err(err) = res {
            assert!(matches!(err, AvifError::InvalidImageGrid(_)));
        }
        Ok(())
    }
}
