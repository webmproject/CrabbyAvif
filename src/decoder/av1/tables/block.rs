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

//! Block, transform and context lookup tables (spec sections 5.11, 8.3.2
//! and 9.3).
//!
//! Tables whose spec element type is a block size, transform size,
//! transform type or prediction mode store the corresponding enum from
//! [`crate::decoder::av1::types`]. Tables are indexed with `enum as usize`.

use crate::decoder::av1::types::{BlockSize, IntraMode, TxSize, TxType};

/// Spec `Block_Width[ x ]` = `4 * Num_4x4_Blocks_Wide[ x ]` (section 9.3),
/// the block width in samples.
pub const BLOCK_WIDTH: [u8; 22] = {
    let mut t = [0u8; 22];
    let mut i = 0;
    while i < 22 {
        t[i] = NUM_4X4_BLOCKS_WIDE[i] * 4;
        i += 1;
    }
    t
};

/// Spec `Block_Height[ x ]` = `4 * Num_4x4_Blocks_High[ x ]` (section 9.3),
/// the block height in samples.
pub const BLOCK_HEIGHT: [u8; 22] = {
    let mut t = [0u8; 22];
    let mut i = 0;
    while i < 22 {
        t[i] = NUM_4X4_BLOCKS_HIGH[i] * 4;
        i += 1;
    }
    t
};

// BEGIN GENERATED (extracted mechanically from the spec text)

/// `Mi_Width_Log2[BLOCK_SIZES]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const MI_WIDTH_LOG2: [u8; 22] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4,
];

/// `Mi_Height_Log2[BLOCK_SIZES]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const MI_HEIGHT_LOG2: [u8; 22] = [
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2,
];

/// `Num_4x4_Blocks_Wide[BLOCK_SIZES]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const NUM_4X4_BLOCKS_WIDE: [u8; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];

/// `Num_4x4_Blocks_High[BLOCK_SIZES]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const NUM_4X4_BLOCKS_HIGH: [u8; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// `Size_Group[BLOCK_SIZES]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const SIZE_GROUP: [u8; 22] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2];

/// `Max_Tx_Size_Rect[BLOCK_SIZES]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const MAX_TX_SIZE_RECT: [TxSize; 22] = [
    TxSize::Tx4x4, TxSize::Tx4x8, TxSize::Tx8x4, TxSize::Tx8x8, TxSize::Tx8x16, TxSize::Tx16x8,
    TxSize::Tx16x16, TxSize::Tx16x32, TxSize::Tx32x16, TxSize::Tx32x32, TxSize::Tx32x64,
    TxSize::Tx64x32, TxSize::Tx64x64, TxSize::Tx64x64, TxSize::Tx64x64, TxSize::Tx64x64,
    TxSize::Tx4x16, TxSize::Tx16x4, TxSize::Tx8x32, TxSize::Tx32x8, TxSize::Tx16x64,
    TxSize::Tx64x16,
];

/// `Max_Tx_Depth[BLOCK_SIZES]` (spec section 5.11.15, TX size syntax).
#[rustfmt::skip]
pub const MAX_TX_DEPTH: [u8; 22] = [
    0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4,
];

/// `Partition_Subsize[10][BLOCK_SIZES]` (spec section 9.3, Conversion tables).
/// Indexed by `[partition][block_size]`. `None` is the spec's
/// `BLOCK_INVALID` (only reached for rectangular block sizes, which
/// the spec never looks up).
#[rustfmt::skip]
pub static PARTITION_SUBSIZE: [[Option<BlockSize>; 22]; 10] = [
    [
        Some(BlockSize::B4x4), None, None, Some(BlockSize::B8x8), None, None,
        Some(BlockSize::B16x16), None, None, Some(BlockSize::B32x32), None, None,
        Some(BlockSize::B64x64), None, None, Some(BlockSize::B128x128), None, None, None, None,
        None, None,
    ],
    [
        None, None, None, Some(BlockSize::B8x4), None, None, Some(BlockSize::B16x8), None, None,
        Some(BlockSize::B32x16), None, None, Some(BlockSize::B64x32), None, None,
        Some(BlockSize::B128x64), None, None, None, None, None, None,
    ],
    [
        None, None, None, Some(BlockSize::B4x8), None, None, Some(BlockSize::B8x16), None, None,
        Some(BlockSize::B16x32), None, None, Some(BlockSize::B32x64), None, None,
        Some(BlockSize::B64x128), None, None, None, None, None, None,
    ],
    [
        None, None, None, Some(BlockSize::B4x4), None, None, Some(BlockSize::B8x8), None, None,
        Some(BlockSize::B16x16), None, None, Some(BlockSize::B32x32), None, None,
        Some(BlockSize::B64x64), None, None, None, None, None, None,
    ],
    [
        None, None, None, Some(BlockSize::B8x4), None, None, Some(BlockSize::B16x8), None, None,
        Some(BlockSize::B32x16), None, None, Some(BlockSize::B64x32), None, None,
        Some(BlockSize::B128x64), None, None, None, None, None, None,
    ],
    [
        None, None, None, Some(BlockSize::B8x4), None, None, Some(BlockSize::B16x8), None, None,
        Some(BlockSize::B32x16), None, None, Some(BlockSize::B64x32), None, None,
        Some(BlockSize::B128x64), None, None, None, None, None, None,
    ],
    [
        None, None, None, Some(BlockSize::B4x8), None, None, Some(BlockSize::B8x16), None, None,
        Some(BlockSize::B16x32), None, None, Some(BlockSize::B32x64), None, None,
        Some(BlockSize::B64x128), None, None, None, None, None, None,
    ],
    [
        None, None, None, Some(BlockSize::B4x8), None, None, Some(BlockSize::B8x16), None, None,
        Some(BlockSize::B16x32), None, None, Some(BlockSize::B32x64), None, None,
        Some(BlockSize::B64x128), None, None, None, None, None, None,
    ],
    [
        None, None, None, None, None, None, Some(BlockSize::B16x4), None, None,
        Some(BlockSize::B32x8), None, None, Some(BlockSize::B64x16), None, None, None, None, None,
        None, None, None, None,
    ],
    [
        None, None, None, None, None, None, Some(BlockSize::B4x16), None, None,
        Some(BlockSize::B8x32), None, None, Some(BlockSize::B16x64), None, None, None, None, None,
        None, None, None, None,
    ],
];

/// `Subsampled_Size[BLOCK_SIZES][2][2]` (spec section 5.11.38, Get plane residual size function).
/// Indexed by `[block_size][subsampling_x][subsampling_y]`. `None`
/// is the spec's `BLOCK_INVALID` (a block size / subsampling
/// combination that cannot occur in a conformant stream).
#[rustfmt::skip]
pub static SUBSAMPLED_SIZE: [[[Option<BlockSize>; 2]; 2]; 22] = [
    [
        [Some(BlockSize::B4x4), Some(BlockSize::B4x4)],
        [Some(BlockSize::B4x4), Some(BlockSize::B4x4)],
    ],
    [
        [Some(BlockSize::B4x8), Some(BlockSize::B4x4)],
        [None, Some(BlockSize::B4x4)],
    ],
    [
        [Some(BlockSize::B8x4), None],
        [Some(BlockSize::B4x4), Some(BlockSize::B4x4)],
    ],
    [
        [Some(BlockSize::B8x8), Some(BlockSize::B8x4)],
        [Some(BlockSize::B4x8), Some(BlockSize::B4x4)],
    ],
    [
        [Some(BlockSize::B8x16), Some(BlockSize::B8x8)],
        [None, Some(BlockSize::B4x8)],
    ],
    [
        [Some(BlockSize::B16x8), None],
        [Some(BlockSize::B8x8), Some(BlockSize::B8x4)],
    ],
    [
        [Some(BlockSize::B16x16), Some(BlockSize::B16x8)],
        [Some(BlockSize::B8x16), Some(BlockSize::B8x8)],
    ],
    [
        [Some(BlockSize::B16x32), Some(BlockSize::B16x16)],
        [None, Some(BlockSize::B8x16)],
    ],
    [
        [Some(BlockSize::B32x16), None],
        [Some(BlockSize::B16x16), Some(BlockSize::B16x8)],
    ],
    [
        [Some(BlockSize::B32x32), Some(BlockSize::B32x16)],
        [Some(BlockSize::B16x32), Some(BlockSize::B16x16)],
    ],
    [
        [Some(BlockSize::B32x64), Some(BlockSize::B32x32)],
        [None, Some(BlockSize::B16x32)],
    ],
    [
        [Some(BlockSize::B64x32), None],
        [Some(BlockSize::B32x32), Some(BlockSize::B32x16)],
    ],
    [
        [Some(BlockSize::B64x64), Some(BlockSize::B64x32)],
        [Some(BlockSize::B32x64), Some(BlockSize::B32x32)],
    ],
    [
        [Some(BlockSize::B64x128), Some(BlockSize::B64x64)],
        [None, Some(BlockSize::B32x64)],
    ],
    [
        [Some(BlockSize::B128x64), None],
        [Some(BlockSize::B64x64), Some(BlockSize::B64x32)],
    ],
    [
        [Some(BlockSize::B128x128), Some(BlockSize::B128x64)],
        [Some(BlockSize::B64x128), Some(BlockSize::B64x64)],
    ],
    [
        [Some(BlockSize::B4x16), Some(BlockSize::B4x8)],
        [None, Some(BlockSize::B4x8)],
    ],
    [
        [Some(BlockSize::B16x4), None],
        [Some(BlockSize::B8x4), Some(BlockSize::B8x4)],
    ],
    [
        [Some(BlockSize::B8x32), Some(BlockSize::B8x16)],
        [None, Some(BlockSize::B4x16)],
    ],
    [
        [Some(BlockSize::B32x8), None],
        [Some(BlockSize::B16x8), Some(BlockSize::B16x4)],
    ],
    [
        [Some(BlockSize::B16x64), Some(BlockSize::B16x32)],
        [None, Some(BlockSize::B8x32)],
    ],
    [
        [Some(BlockSize::B64x16), None],
        [Some(BlockSize::B32x16), Some(BlockSize::B32x8)],
    ],
];

/// `Split_Tx_Size[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const SPLIT_TX_SIZE: [TxSize; 19] = [
    TxSize::Tx4x4, TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32, TxSize::Tx4x4,
    TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx16x16, TxSize::Tx32x32,
    TxSize::Tx32x32, TxSize::Tx4x8, TxSize::Tx8x4, TxSize::Tx8x16, TxSize::Tx16x8, TxSize::Tx16x32,
    TxSize::Tx32x16,
];

/// `Tx_Width[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const TX_WIDTH: [u8; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];

/// `Tx_Height[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const TX_HEIGHT: [u8; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];

/// `Tx_Width_Log2[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const TX_WIDTH_LOG2: [u8; 19] = [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];

/// `Tx_Height_Log2[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const TX_HEIGHT_LOG2: [u8; 19] = [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];

/// `Tx_Size_Sqr[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const TX_SIZE_SQR: [TxSize; 19] = [
    TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32, TxSize::Tx64x64, TxSize::Tx4x4,
    TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx16x16, TxSize::Tx32x32,
    TxSize::Tx32x32, TxSize::Tx4x4, TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx8x8, TxSize::Tx16x16,
    TxSize::Tx16x16,
];

/// `Tx_Size_Sqr_Up[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const TX_SIZE_SQR_UP: [TxSize; 19] = [
    TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32, TxSize::Tx64x64, TxSize::Tx8x8,
    TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx16x16, TxSize::Tx32x32, TxSize::Tx32x32,
    TxSize::Tx64x64, TxSize::Tx64x64, TxSize::Tx16x16, TxSize::Tx16x16, TxSize::Tx32x32,
    TxSize::Tx32x32, TxSize::Tx64x64, TxSize::Tx64x64,
];

/// `Adjusted_Tx_Size[TX_SIZES_ALL]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const ADJUSTED_TX_SIZE: [TxSize; 19] = [
    TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32, TxSize::Tx32x32, TxSize::Tx4x8,
    TxSize::Tx8x4, TxSize::Tx8x16, TxSize::Tx16x8, TxSize::Tx16x32, TxSize::Tx32x16,
    TxSize::Tx32x32, TxSize::Tx32x32, TxSize::Tx4x16, TxSize::Tx16x4, TxSize::Tx8x32,
    TxSize::Tx32x8, TxSize::Tx16x32, TxSize::Tx32x16,
];

/// `Mode_To_Txfm[UV_INTRA_MODES_CFL_ALLOWED]` (spec section 9.3, Conversion tables).
/// Indexed by the (uv) intra mode, including `UV_CFL_PRED`.
#[rustfmt::skip]
pub const MODE_TO_TXFM: [TxType; 14] = [
    TxType::DCT_DCT, TxType::ADST_DCT, TxType::DCT_ADST, TxType::DCT_DCT, TxType::ADST_ADST,
    TxType::ADST_DCT, TxType::DCT_ADST, TxType::DCT_ADST, TxType::ADST_DCT, TxType::ADST_ADST,
    TxType::ADST_DCT, TxType::DCT_ADST, TxType::ADST_ADST, TxType::DCT_DCT,
];

/// `Tx_Type_In_Set_Intra[TX_SET_TYPES_INTRA][TX_TYPES]`
/// (spec section 5.11.40, Compute transform type function).
/// Indexed by `[tx_set][tx_type]`; `true` if the type is allowed.
#[rustfmt::skip]
pub static TX_TYPE_IN_SET_INTRA: [[bool; 16]; 3] = [
    [
        true, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false,
    ],
    [
        true, true, true, true, false, false, false, false, false, true, true, true, false, false,
        false, false,
    ],
    [
        true, true, true, true, false, false, false, false, false, true, false, false, false, false,
        false, false,
    ],
];

/// `Tx_Type_In_Set_Inter[TX_SET_TYPES_INTER][TX_TYPES]`
/// (spec section 5.11.40, Compute transform type function).
/// Indexed by `[tx_set][tx_type]`; `true` if the type is allowed.
#[rustfmt::skip]
pub static TX_TYPE_IN_SET_INTER: [[bool; 16]; 4] = [
    [
        true, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false,
    ],
    [
        true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
        true,
    ],
    [
        true, true, true, true, true, true, true, true, true, true, true, true, false, false, false,
        false,
    ],
    [
        true, false, false, false, false, false, false, false, false, true, false, false, false,
        false, false, false,
    ],
];

/// `Tx_Type_Intra_Inv_Set1[7]` (spec section 5.11.47, Transform type syntax).
#[rustfmt::skip]
pub const TX_TYPE_INTRA_INV_SET1: [TxType; 7] = [
    TxType::IDTX, TxType::DCT_DCT, TxType::V_DCT, TxType::H_DCT, TxType::ADST_ADST,
    TxType::ADST_DCT, TxType::DCT_ADST,
];

/// `Tx_Type_Intra_Inv_Set2[5]` (spec section 5.11.47, Transform type syntax).
#[rustfmt::skip]
pub const TX_TYPE_INTRA_INV_SET2: [TxType; 5] = [
    TxType::IDTX, TxType::DCT_DCT, TxType::ADST_ADST, TxType::ADST_DCT, TxType::DCT_ADST,
];

/// `Tx_Type_Inter_Inv_Set1[16]` (spec section 5.11.47, Transform type syntax).
#[rustfmt::skip]
pub const TX_TYPE_INTER_INV_SET1: [TxType; 16] = [
    TxType::IDTX, TxType::V_DCT, TxType::H_DCT, TxType::V_ADST, TxType::H_ADST, TxType::V_FLIPADST,
    TxType::H_FLIPADST, TxType::DCT_DCT, TxType::ADST_DCT, TxType::DCT_ADST, TxType::FLIPADST_DCT,
    TxType::DCT_FLIPADST, TxType::ADST_ADST, TxType::FLIPADST_FLIPADST, TxType::ADST_FLIPADST,
    TxType::FLIPADST_ADST,
];

/// `Tx_Type_Inter_Inv_Set2[12]` (spec section 5.11.47, Transform type syntax).
#[rustfmt::skip]
pub const TX_TYPE_INTER_INV_SET2: [TxType; 12] = [
    TxType::IDTX, TxType::V_DCT, TxType::H_DCT, TxType::DCT_DCT, TxType::ADST_DCT, TxType::DCT_ADST,
    TxType::FLIPADST_DCT, TxType::DCT_FLIPADST, TxType::ADST_ADST, TxType::FLIPADST_FLIPADST,
    TxType::ADST_FLIPADST, TxType::FLIPADST_ADST,
];

/// `Tx_Type_Inter_Inv_Set3[2]` (spec section 5.11.47, Transform type syntax).
#[rustfmt::skip]
pub const TX_TYPE_INTER_INV_SET3: [TxType; 2] = [TxType::IDTX, TxType::DCT_DCT];

/// `Filter_Intra_Mode_To_Intra_Dir[INTRA_FILTER_MODES]`
/// (spec section 8.3.2, Cdf selection process).
#[rustfmt::skip]
pub const FILTER_INTRA_MODE_TO_INTRA_DIR: [IntraMode; 5] = [
    IntraMode::DC_PRED, IntraMode::V_PRED, IntraMode::H_PRED, IntraMode::D157_PRED,
    IntraMode::DC_PRED,
];

/// `Intra_Mode_Context[INTRA_MODES]` (spec section 8.3.2, Cdf selection process).
#[rustfmt::skip]
pub const INTRA_MODE_CONTEXT: [u8; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `Palette_Color_Context[PALETTE_MAX_COLOR_CONTEXT_HASH + 1]`
/// (spec section 9.3, Conversion tables).
/// Negative entries are never accessed.
#[rustfmt::skip]
pub const PALETTE_COLOR_CONTEXT: [i8; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];

/// `Palette_Color_Hash_Multipliers[PALETTE_NUM_NEIGHBORS]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub const PALETTE_COLOR_HASH_MULTIPLIERS: [u8; 3] = [1, 2, 2];

/// `Coeff_Base_Ctx_Offset[TX_SIZES_ALL][5][5]` (spec section 8.3.2, Cdf selection process).
#[rustfmt::skip]
pub static COEFF_BASE_CTX_OFFSET: [[[u8; 5]; 5]; 19] = [
    [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
];

/// `Coeff_Base_Pos_Ctx_Offset[3]` (spec section 8.3.2, Cdf selection process).
#[rustfmt::skip]
pub const COEFF_BASE_POS_CTX_OFFSET: [u8; 3] = [26, 31, 36];

/// `Mag_Ref_Offset_With_Tx_Class[3][3][2]` (spec section 8.3.2, Cdf selection process).
#[rustfmt::skip]
pub static MAG_REF_OFFSET_WITH_TX_CLASS: [[[u8; 2]; 3]; 3] = [
    [
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    [
        [0, 1],
        [1, 0],
        [0, 2],
    ],
    [
        [0, 1],
        [1, 0],
        [2, 0],
    ],
];

/// `Sig_Ref_Diff_Offset[3][SIG_REF_DIFF_OFFSET_NUM][2]` (spec section 9.3, Conversion tables).
#[rustfmt::skip]
pub static SIG_REF_DIFF_OFFSET: [[[u8; 2]; 5]; 3] = [
    [
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [2, 0],
    ],
    [
        [0, 1],
        [1, 0],
        [0, 2],
        [0, 3],
        [0, 4],
    ],
    [
        [0, 1],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
    ],
];
// END GENERATED

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::types::Partition;

    /// (width, height) parsed from the enum variant name, e.g. `B16x64`.
    fn dims<T: core::fmt::Debug>(v: T, prefix: &str) -> (usize, usize) {
        let s = format!("{v:?}");
        let (w, h) = s.strip_prefix(prefix).unwrap().split_once('x').unwrap();
        (w.parse().unwrap(), h.parse().unwrap())
    }

    fn bdims(b: BlockSize) -> (usize, usize) {
        dims(b, "B")
    }

    fn tdims(t: TxSize) -> (usize, usize) {
        dims(t, "Tx")
    }

    #[test]
    fn block_geometry() {
        for &b in BlockSize::ALL {
            let i = b as usize;
            let (w, h) = bdims(b);
            assert_eq!(BLOCK_WIDTH[i] as usize, w);
            assert_eq!(BLOCK_HEIGHT[i] as usize, h);
            assert_eq!(NUM_4X4_BLOCKS_WIDE[i] as usize * 4, w);
            assert_eq!(NUM_4X4_BLOCKS_HIGH[i] as usize * 4, h);
            assert_eq!(4 << MI_WIDTH_LOG2[i], w);
            assert_eq!(4 << MI_HEIGHT_LOG2[i], h);
            let (tw, th) = tdims(MAX_TX_SIZE_RECT[i]);
            assert_eq!((tw, th), (w.min(64), h.min(64)), "{b:?}");
        }
    }

    #[test]
    fn tx_geometry() {
        for &t in TxSize::ALL {
            let i = t as usize;
            let (w, h) = tdims(t);
            assert_eq!(TX_WIDTH[i] as usize, w);
            assert_eq!(TX_HEIGHT[i] as usize, h);
            assert_eq!(1 << TX_WIDTH_LOG2[i], w);
            assert_eq!(1 << TX_HEIGHT_LOG2[i], h);
            let m = w.min(h);
            let x = w.max(h);
            assert_eq!(tdims(TX_SIZE_SQR[i]), (m, m));
            assert_eq!(tdims(TX_SIZE_SQR_UP[i]), (x, x));
            assert_eq!(tdims(ADJUSTED_TX_SIZE[i]), (w.min(32), h.min(32)));
            let split = match w.cmp(&h) {
                core::cmp::Ordering::Equal => ((w / 2).max(4), (h / 2).max(4)),
                core::cmp::Ordering::Greater => (w / 2, h),
                core::cmp::Ordering::Less => (w, h / 2),
            };
            assert_eq!(tdims(SPLIT_TX_SIZE[i]), split, "{t:?}");
        }
    }

    #[test]
    fn partition_subsize() {
        for &b in BlockSize::ALL {
            let (w, h) = bdims(b);
            for &p in Partition::ALL {
                let s = PARTITION_SUBSIZE[p as usize][b as usize];
                if w != h {
                    assert_eq!(s, None);
                    continue;
                }
                let expect = match p {
                    Partition::None => Some((w, h)),
                    Partition::Horz | Partition::HorzA | Partition::HorzB => Some((w, h / 2)),
                    Partition::Vert | Partition::VertA | Partition::VertB => Some((w / 2, h)),
                    Partition::Split => Some((w / 2, h / 2)),
                    Partition::Horz4 => Some((w, h / 4)),
                    Partition::Vert4 => Some((w / 4, h)),
                };
                let valid = |(sw, sh): (usize, usize)| {
                    BlockSize::ALL.iter().any(|&x| bdims(x) == (sw, sh))
                        && !(matches!(p, Partition::Horz4 | Partition::Vert4) && w == 128)
                };
                assert_eq!(s.map(bdims), expect.filter(|&d| valid(d)), "{p:?} {b:?}");
            }
        }
    }

    #[test]
    fn subsampled_size() {
        for &b in BlockSize::ALL {
            let (w, h) = bdims(b);
            for (ssx, row) in SUBSAMPLED_SIZE[b as usize].iter().enumerate() {
                for (ssy, &s) in row.iter().enumerate() {
                    if let Some(s) = s {
                        assert_eq!(bdims(s), ((w >> ssx).max(4), (h >> ssy).max(4)));
                    } else {
                        // Only 4:2:2 / 4:4:0 style combinations producing
                        // a 2 sample dimension (or a 4:1 aspect ratio
                        // that has no block size) are invalid.
                        assert!(ssx != ssy, "{b:?} {ssx} {ssy}");
                    }
                }
            }
        }
    }

    #[test]
    fn tx_type_sets() {
        let check = |inv: &[TxType], set: &[bool; 16]| {
            assert_eq!(inv.len(), set.iter().filter(|&&x| x).count());
            for &t in inv {
                assert!(set[t as usize]);
            }
        };
        check(&TX_TYPE_INTRA_INV_SET1, &TX_TYPE_IN_SET_INTRA[1]);
        check(&TX_TYPE_INTRA_INV_SET2, &TX_TYPE_IN_SET_INTRA[2]);
        check(&TX_TYPE_INTER_INV_SET1, &TX_TYPE_IN_SET_INTER[1]);
        check(&TX_TYPE_INTER_INV_SET2, &TX_TYPE_IN_SET_INTER[2]);
        check(&TX_TYPE_INTER_INV_SET3, &TX_TYPE_IN_SET_INTER[3]);
        assert_eq!(TX_TYPE_IN_SET_INTRA[0].iter().filter(|&&x| x).count(), 1);
        assert_eq!(TX_TYPE_IN_SET_INTER[0].iter().filter(|&&x| x).count(), 1);
    }

    #[test]
    fn spot_check_values() {
        use IntraMode::*;
        assert_eq!(MODE_TO_TXFM[V_PRED as usize], TxType::ADST_DCT);
        assert_eq!(MODE_TO_TXFM[H_PRED as usize], TxType::DCT_ADST);
        assert_eq!(MODE_TO_TXFM[UV_CFL_PRED as usize], TxType::DCT_DCT);
        assert_eq!(
            FILTER_INTRA_MODE_TO_INTRA_DIR,
            [DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED]
        );
        assert_eq!(PALETTE_COLOR_CONTEXT, [-1, -1, 0, -1, -1, 4, 3, 2, 1]);
        assert_eq!(PALETTE_COLOR_HASH_MULTIPLIERS, [1, 2, 2]);
        assert_eq!(COEFF_BASE_POS_CTX_OFFSET, [26, 31, 36]);
        assert_eq!(MAX_TX_DEPTH[BlockSize::B64x64 as usize], 4);
        assert_eq!(SIZE_GROUP[BlockSize::B16x64 as usize], 2);
        assert_eq!(
            COEFF_BASE_CTX_OFFSET[TxSize::Tx4x4 as usize][0],
            [0, 1, 6, 6, 0]
        );
        assert_eq!(MAG_REF_OFFSET_WITH_TX_CLASS[0], [[0, 1], [1, 0], [1, 1]]);
        assert_eq!(SIG_REF_DIFF_OFFSET[0][0], [0, 1]);
        assert_eq!(INTRA_MODE_CONTEXT, [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0]);
    }
}
