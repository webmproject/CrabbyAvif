use super::image::*;
use super::io::*;
use super::types::*;

use crate::decoder::gainmap::*;
use crate::internal_utils::*;
use crate::parser::mp4box::*;
use crate::*;

pub type avifContentLightLevelInformationBox = ContentLightLevelInformation;

#[repr(C)]
#[derive(Debug, Default)]
pub struct avifGainMapMetadata {
    pub gainMapMinN: [i32; 3],
    pub gainMapMinD: [u32; 3],
    pub gainMapMaxN: [i32; 3],
    pub gainMapMaxD: [u32; 3],
    pub gainMapGammaN: [u32; 3],
    pub gainMapGammaD: [u32; 3],
    pub baseOffsetN: [i32; 3],
    pub baseOffsetD: [u32; 3],
    pub alternateOffsetN: [i32; 3],
    pub alternateOffsetD: [u32; 3],
    pub baseHdrHeadroomN: u32,
    pub baseHdrHeadroomD: u32,
    pub alternateHdrHeadroomN: u32,
    pub alternateHdrHeadroomD: u32,
    pub backwardDirection: avifBool,
    pub useBaseColorSpace: avifBool,
}

impl Fraction {
    pub fn n_u32(self) -> u32 {
        if self.is_negative {
            panic!()
        } else {
            self.n
        }
    }
}

impl From<&GainMapMetadata> for avifGainMapMetadata {
    fn from(m: &GainMapMetadata) -> Self {
        avifGainMapMetadata {
            gainMapMinN: [m.min[0].n_i32(), m.min[1].n_i32(), m.min[2].n_i32()],
            gainMapMinD: [m.min[0].d, m.min[1].d, m.min[2].d],
            gainMapMaxN: [m.max[0].n_i32(), m.max[1].n_i32(), m.max[2].n_i32()],
            gainMapMaxD: [m.max[0].d, m.max[1].d, m.max[2].d],
            gainMapGammaN: [m.gamma[0].n_u32(), m.gamma[1].n_u32(), m.gamma[2].n_u32()],
            gainMapGammaD: [m.gamma[0].d, m.gamma[1].d, m.gamma[2].d],
            baseOffsetN: [
                m.base_offset[0].n_i32(),
                m.base_offset[1].n_i32(),
                m.base_offset[2].n_i32(),
            ],
            baseOffsetD: [m.base_offset[0].d, m.base_offset[1].d, m.base_offset[2].d],
            alternateOffsetN: [
                m.alternate_offset[0].n_i32(),
                m.alternate_offset[1].n_i32(),
                m.alternate_offset[2].n_i32(),
            ],
            alternateOffsetD: [
                m.alternate_offset[0].d,
                m.alternate_offset[1].d,
                m.alternate_offset[2].d,
            ],
            baseHdrHeadroomN: m.base_hdr_headroom.n_u32(),
            baseHdrHeadroomD: m.base_hdr_headroom.d,
            alternateHdrHeadroomN: m.alternate_hdr_headroom.n_u32(),
            alternateHdrHeadroomD: m.alternate_hdr_headroom.d,
            backwardDirection: m.backward_direction as avifBool,
            useBaseColorSpace: m.use_base_color_space as avifBool,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct avifGainMap {
    pub image: *mut avifImage,
    pub metadata: avifGainMapMetadata,
    pub altICC: avifRWData,
    pub altColorPrimaries: ColorPrimaries,
    pub altTransferCharacteristics: TransferCharacteristics,
    pub altMatrixCoefficients: MatrixCoefficients,
    pub altYUVRange: avifRange,
    pub altDepth: u32,
    pub altPlaneCount: u32,
    pub altCLLI: avifContentLightLevelInformationBox,
}

impl Default for avifGainMap {
    fn default() -> Self {
        avifGainMap {
            image: std::ptr::null_mut(),
            metadata: avifGainMapMetadata::default(),
            altICC: avifRWData::default(),
            altColorPrimaries: ColorPrimaries::default(),
            altTransferCharacteristics: TransferCharacteristics::default(),
            altMatrixCoefficients: MatrixCoefficients::default(),
            altYUVRange: avifRange::Full,
            altDepth: 0,
            altPlaneCount: 0,
            altCLLI: Default::default(),
        }
    }
}

impl From<&GainMap> for avifGainMap {
    fn from(gainmap: &GainMap) -> Self {
        avifGainMap {
            metadata: (&gainmap.metadata).into(),
            altICC: (&gainmap.alt_icc).into(),
            altColorPrimaries: gainmap.alt_color_primaries,
            altTransferCharacteristics: gainmap.alt_transfer_characteristics,
            altMatrixCoefficients: gainmap.alt_matrix_coefficients,
            altYUVRange: gainmap.alt_full_range.into(),
            altDepth: u32::from(gainmap.alt_plane_depth),
            altPlaneCount: u32::from(gainmap.alt_plane_count),
            altCLLI: gainmap.alt_clli,
            ..Self::default()
        }
    }
}
