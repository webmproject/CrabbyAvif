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

//! Core enums shared across the decoder. Variant order matches the AV1
//! specification so `as usize` can be used to index spec tables.

#![allow(non_camel_case_types)]

macro_rules! spec_enum {
    ($(#[$m:meta])* $name:ident { $($v:ident),* $(,)? }) => {
        $(#[$m])*
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
        #[repr(u8)]
        pub enum $name {
            #[default]
            $($v),*
        }
        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$v),*];
            pub const COUNT: usize = Self::ALL.len();
            /// Converts a spec value into the enum. Panics if out of range
            /// (callers only pass decoded symbols that are in range).
            #[inline(always)]
            pub fn from_u8(v: u8) -> Self {
                Self::ALL[v as usize]
            }
            #[inline(always)]
            pub fn idx(self) -> usize {
                self as usize
            }
        }
    };
}

// The default variant is the first one listed.
spec_enum!(
    /// Block sizes (spec `BLOCK_*`).
    BlockSize {
        B4x4,
        B4x8,
        B8x4,
        B8x8,
        B8x16,
        B16x8,
        B16x16,
        B16x32,
        B32x16,
        B32x32,
        B32x64,
        B64x32,
        B64x64,
        B64x128,
        B128x64,
        B128x128,
        B4x16,
        B16x4,
        B8x32,
        B32x8,
        B16x64,
        B64x16,
    }
);

spec_enum!(
    /// Transform sizes (spec `TX_*`).
    TxSize {
        Tx4x4,
        Tx8x8,
        Tx16x16,
        Tx32x32,
        Tx64x64,
        Tx4x8,
        Tx8x4,
        Tx8x16,
        Tx16x8,
        Tx16x32,
        Tx32x16,
        Tx32x64,
        Tx64x32,
        Tx4x16,
        Tx16x4,
        Tx8x32,
        Tx32x8,
        Tx16x64,
        Tx64x16,
    }
);

spec_enum!(
    /// Transform types (spec order). The first part of the name is the
    /// vertical (column) transform, the second the horizontal (row) one.
    TxType {
        DCT_DCT,
        ADST_DCT,
        DCT_ADST,
        ADST_ADST,
        FLIPADST_DCT,
        DCT_FLIPADST,
        FLIPADST_FLIPADST,
        ADST_FLIPADST,
        FLIPADST_ADST,
        IDTX,
        V_DCT,
        H_DCT,
        V_ADST,
        H_ADST,
        V_FLIPADST,
        H_FLIPADST,
    }
);

spec_enum!(
    /// Partition types (spec `PARTITION_*`).
    Partition {
        None,
        Horz,
        Vert,
        Split,
        HorzA,
        HorzB,
        VertA,
        VertB,
        Horz4,
        Vert4,
    }
);

spec_enum!(
    /// Intra prediction modes (spec `*_PRED`). `UV_CFL_PRED` is only valid
    /// for chroma.
    IntraMode {
        DC_PRED,
        V_PRED,
        H_PRED,
        D45_PRED,
        D135_PRED,
        D113_PRED,
        D157_PRED,
        D203_PRED,
        D67_PRED,
        SMOOTH_PRED,
        SMOOTH_V_PRED,
        SMOOTH_H_PRED,
        PAETH_PRED,
        UV_CFL_PRED,
    }
);

spec_enum!(
    /// Filter intra modes (spec `FILTER_*_PRED`).
    FilterIntraMode {
        FILTER_DC_PRED,
        FILTER_V_PRED,
        FILTER_H_PRED,
        FILTER_D157_PRED,
        FILTER_PAETH_PRED,
    }
);

impl IntraMode {
    /// Spec `is_directional_mode`.
    #[inline]
    pub fn is_directional(self) -> bool {
        (IntraMode::V_PRED as u8..=IntraMode::D67_PRED as u8).contains(&(self as u8))
    }
}

pub const MI_SIZE: usize = 4;
pub const MI_SIZE_LOG2: usize = 2;
pub const MAX_SB_SIZE: usize = 128;
pub const MAX_SB_SIZE_LOG2: usize = 7;
pub const MAX_TILE_COLS: usize = 64;
pub const MAX_TILE_ROWS: usize = 64;
pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_MAX: usize = 8;
pub const SEG_LVL_ALT_Q: usize = 0;
pub const SEG_LVL_ALT_LF_Y_V: usize = 1;
pub const SEG_LVL_REF_FRAME: usize = 5;
pub const SEG_LVL_SKIP: usize = 6;
pub const SEG_LVL_GLOBALMV: usize = 7;
pub const PALETTE_COLORS: usize = 8;
pub const PALETTE_NUM_NEIGHBORS: usize = 3;
pub const DELTA_Q_SMALL: u32 = 3;
pub const DELTA_LF_SMALL: u32 = 3;
pub const FRAME_LF_COUNT: usize = 4;
pub const MAX_LOOP_FILTER: i32 = 63;
pub const RESTORATION_TILESIZE_MAX: usize = 256;
pub const SUPERRES_NUM: u32 = 8;
pub const SUPERRES_DENOM_MIN: u32 = 9;
pub const SUPERRES_DENOM_BITS: u32 = 3;
pub const SUPERRES_FILTER_BITS: u32 = 6;
pub const SUPERRES_FILTER_TAPS: usize = 8;
pub const SUPERRES_FILTER_SHIFTS: u32 = 1 << SUPERRES_FILTER_BITS;
pub const SUPERRES_EXTRA_BITS: u32 = 8;
pub const SUPERRES_SCALE_BITS: u32 = 14;
pub const SUPERRES_SCALE_MASK: u32 = (1 << 14) - 1;
