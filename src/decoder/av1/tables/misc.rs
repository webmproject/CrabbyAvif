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

//! Frame header / loop restoration related tables (spec sections 5.9 and
//! 5.11).

/// `FrameRestorationType` values (spec section 6.10.15).
pub const RESTORE_NONE: u8 = 0;
/// `FrameRestorationType` values (spec section 6.10.15).
pub const RESTORE_WIENER: u8 = 1;
/// `FrameRestorationType` values (spec section 6.10.15).
pub const RESTORE_SGRPROJ: u8 = 2;
/// `FrameRestorationType` values (spec section 6.10.15).
pub const RESTORE_SWITCHABLE: u8 = 3;

// BEGIN GENERATED (extracted mechanically from the spec text)

/// `Segmentation_Feature_Bits[SEG_LVL_MAX]` (spec section 5.9.14, Segmentation params syntax).
#[rustfmt::skip]
pub const SEGMENTATION_FEATURE_BITS: [u8; 8] = [8, 6, 6, 6, 6, 3, 0, 0];

/// `Segmentation_Feature_Signed[SEG_LVL_MAX]` (spec section 5.9.14, Segmentation params syntax).
#[rustfmt::skip]
pub const SEGMENTATION_FEATURE_SIGNED: [bool; 8] = [
    true, true, true, true, true, false, false, false,
];

/// `Segmentation_Feature_Max[SEG_LVL_MAX]` (spec section 5.9.14, Segmentation params syntax).
#[rustfmt::skip]
pub const SEGMENTATION_FEATURE_MAX: [u8; 8] = [255, 63, 63, 63, 63, 7, 0, 0];

/// `Remap_Lr_Type[4]` (spec section 5.9.20, Loop restoration params syntax).
/// Maps `lr_type` to `FrameRestorationType` (`RESTORE_*`).
#[rustfmt::skip]
pub const REMAP_LR_TYPE: [u8; 4] = [
    RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ,
];

/// `Wiener_Taps_Min[3]` (spec section 5.11.58, Read loop restoration unit syntax).
#[rustfmt::skip]
pub const WIENER_TAPS_MIN: [i8; 3] = [-5, -23, -17];

/// `Wiener_Taps_Max[3]` (spec section 5.11.58, Read loop restoration unit syntax).
#[rustfmt::skip]
pub const WIENER_TAPS_MAX: [i8; 3] = [10, 8, 46];

/// `Wiener_Taps_Mid[3]` (spec section 5.11.2, Decode tile syntax).
#[rustfmt::skip]
pub const WIENER_TAPS_MID: [i8; 3] = [3, -7, 15];

/// `Wiener_Taps_K[3]` (spec section 5.11.58, Read loop restoration unit syntax).
#[rustfmt::skip]
pub const WIENER_TAPS_K: [i8; 3] = [1, 2, 3];

/// `Sgrproj_Xqd_Min[2]` (spec section 5.11.58, Read loop restoration unit syntax).
#[rustfmt::skip]
pub const SGRPROJ_XQD_MIN: [i8; 2] = [-96, -32];

/// `Sgrproj_Xqd_Max[2]` (spec section 5.11.58, Read loop restoration unit syntax).
#[rustfmt::skip]
pub const SGRPROJ_XQD_MAX: [i8; 2] = [31, 95];

/// `Sgrproj_Xqd_Mid[2]` (spec section 5.11.2, Decode tile syntax).
#[rustfmt::skip]
pub const SGRPROJ_XQD_MID: [i8; 2] = [-32, 31];
// END GENERATED

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        for i in 0..3 {
            assert!(WIENER_TAPS_MIN[i] <= WIENER_TAPS_MID[i]);
            assert!(WIENER_TAPS_MID[i] <= WIENER_TAPS_MAX[i]);
        }
        for i in 0..2 {
            assert!(SGRPROJ_XQD_MIN[i] <= SGRPROJ_XQD_MID[i]);
            assert!(SGRPROJ_XQD_MID[i] <= SGRPROJ_XQD_MAX[i]);
        }
        assert_eq!(
            REMAP_LR_TYPE,
            [
                RESTORE_NONE,
                RESTORE_SWITCHABLE,
                RESTORE_WIENER,
                RESTORE_SGRPROJ
            ]
        );
        assert_eq!(SEGMENTATION_FEATURE_MAX, [255, 63, 63, 63, 63, 7, 0, 0]);
    }
}
