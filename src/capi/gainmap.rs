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

use super::image::*;
use super::io::*;
use super::types::*;

use crate::decoder::gainmap::*;
use crate::image::YuvRange;
use crate::internal_utils::*;
use crate::parser::mp4box::*;
use crate::*;

pub type avifContentLightLevelInformationBox = ContentLightLevelInformation;

#[repr(C)]
#[derive(Debug)]
pub struct avifSignedFraction {
    pub n: i32,
    pub d: u32,
}

impl From<&Fraction> for avifSignedFraction {
    fn from(fraction: &Fraction) -> Self {
        avifSignedFraction {
            n: fraction.0,
            d: fraction.1,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct avifUnsignedFraction {
    pub n: u32,
    pub d: u32,
}

impl From<&UFraction> for avifUnsignedFraction {
    fn from(fraction: &UFraction) -> Self {
        avifUnsignedFraction {
            n: fraction.0,
            d: fraction.1,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct avifGainMap {
    pub image: *mut avifImage,
    pub gainMapMin: [avifSignedFraction; 3],
    pub gainMapMax: [avifSignedFraction; 3],
    pub gainMapGamma: [avifUnsignedFraction; 3],
    pub baseOffset: [avifSignedFraction; 3],
    pub alternateOffset: [avifSignedFraction; 3],
    pub baseHdrHeadroom: avifUnsignedFraction,
    pub alternateHdrHeadroom: avifUnsignedFraction,
    pub useBaseColorSpace: avifBool,
    pub altICC: avifRWData,
    pub altColorPrimaries: ColorPrimaries,
    pub altTransferCharacteristics: TransferCharacteristics,
    pub altMatrixCoefficients: MatrixCoefficients,
    pub altYUVRange: YuvRange,
    pub altDepth: u32,
    pub altPlaneCount: u32,
    pub altCLLI: avifContentLightLevelInformationBox,
}

impl Default for avifGainMap {
    fn default() -> Self {
        avifGainMap {
            image: std::ptr::null_mut(),
            gainMapMin: [
                avifSignedFraction { n: 1, d: 1 },
                avifSignedFraction { n: 1, d: 1 },
                avifSignedFraction { n: 1, d: 1 },
            ],
            gainMapMax: [
                avifSignedFraction { n: 1, d: 1 },
                avifSignedFraction { n: 1, d: 1 },
                avifSignedFraction { n: 1, d: 1 },
            ],
            gainMapGamma: [
                avifUnsignedFraction { n: 1, d: 1 },
                avifUnsignedFraction { n: 1, d: 1 },
                avifUnsignedFraction { n: 1, d: 1 },
            ],
            baseOffset: [
                avifSignedFraction { n: 1, d: 64 },
                avifSignedFraction { n: 1, d: 64 },
                avifSignedFraction { n: 1, d: 64 },
            ],
            alternateOffset: [
                avifSignedFraction { n: 1, d: 64 },
                avifSignedFraction { n: 1, d: 64 },
                avifSignedFraction { n: 1, d: 64 },
            ],
            baseHdrHeadroom: avifUnsignedFraction { n: 0, d: 1 },
            alternateHdrHeadroom: avifUnsignedFraction { n: 1, d: 1 },
            useBaseColorSpace: to_avifBool(false),
            altICC: avifRWData::default(),
            altColorPrimaries: ColorPrimaries::default(),
            altTransferCharacteristics: TransferCharacteristics::default(),
            altMatrixCoefficients: MatrixCoefficients::default(),
            altYUVRange: YuvRange::Full,
            altDepth: 0,
            altPlaneCount: 0,
            altCLLI: Default::default(),
        }
    }
}

impl From<&GainMap> for avifGainMap {
    fn from(gainmap: &GainMap) -> Self {
        avifGainMap {
            gainMapMin: gainmap.metadata.min.map(|v| (&v).into()),
            gainMapMax: gainmap.metadata.max.map(|v| (&v).into()),
            gainMapGamma: gainmap.metadata.gamma.map(|v| (&v).into()),
            baseOffset: gainmap.metadata.base_offset.map(|v| (&v).into()),
            alternateOffset: gainmap.metadata.alternate_offset.map(|v| (&v).into()),
            baseHdrHeadroom: (&gainmap.metadata.base_hdr_headroom).into(),
            alternateHdrHeadroom: (&gainmap.metadata.alternate_hdr_headroom).into(),
            useBaseColorSpace: gainmap.metadata.use_base_color_space as avifBool,
            altICC: (&gainmap.alt_icc).into(),
            altColorPrimaries: gainmap.alt_color_primaries,
            altTransferCharacteristics: gainmap.alt_transfer_characteristics,
            altMatrixCoefficients: gainmap.alt_matrix_coefficients,
            altYUVRange: gainmap.alt_yuv_range,
            altDepth: u32::from(gainmap.alt_plane_depth),
            altPlaneCount: u32::from(gainmap.alt_plane_count),
            altCLLI: gainmap.alt_clli,
            ..Self::default()
        }
    }
}
