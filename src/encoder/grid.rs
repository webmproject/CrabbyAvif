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

// TODO: b/585625859 - Remove once the helpers are used by Encoder::add_image_split_grid() and
// Encoder::add_image_gainmap_split_grid().
#![allow(dead_code)]

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
}
