use crate::internal_utils::*;
use crate::parser::mp4box::ItemProperty;
use crate::parser::mp4box::MetaBox;
use crate::*;

#[derive(Debug, Default)]
pub struct Track {
    pub id: u32,
    pub aux_for_id: u32,
    pub prem_by_id: u32,
    pub media_timescale: u32,
    pub media_duration: u64,
    pub track_duration: u64,
    pub segment_duration: u64,
    pub is_repeating: bool,
    pub width: u32,
    pub height: u32,
    pub sample_table: Option<SampleTable>,
    pub elst_seen: bool,
    pub meta: Option<MetaBox>,
}

impl Track {
    pub fn is_aux(&self, primary_track_id: u32) -> bool {
        if self.sample_table.is_none() || self.id == 0 {
            return false;
        }
        let sample_table = self.sample_table.as_ref().unwrap();
        if sample_table.chunk_offsets.is_empty() || !sample_table.has_av1_sample() {
            return false;
        }
        self.aux_for_id == primary_track_id
    }

    pub fn is_color(&self) -> bool {
        // If aux_for_id is 0, then it is the color track.
        self.is_aux(0)
    }

    pub fn get_properties(&self) -> Option<&Vec<ItemProperty>> {
        self.sample_table.as_ref()?.get_properties()
    }

    pub fn repetition_count(&self) -> AvifResult<i32> {
        if !self.elst_seen {
            return Ok(-2);
        }
        if self.is_repeating {
            if self.track_duration == u64::MAX {
                // If isRepeating is true and the track duration is unknown/indefinite, then set the
                // repetition count to infinite(Section 9.6.1 of ISO/IEC 23008-12 Part 12).
                return Ok(-1);
            } else {
                // Section 9.6.1. of ISO/IEC 23008-12 Part 12: 1, the entire edit list is repeated a
                // sufficient number of times to equal the track duration.
                //
                // Since libavif uses repetitionCount (which is 0-based), we subtract the value by 1
                // to derive the number of repetitions.
                assert!(self.segment_duration != 0);
                // We specifically check for trackDuration == 0 here and not when it is actually
                // read in order to accept files which inadvertently has a trackDuration of 0
                // without any edit lists.
                if self.track_duration == 0 {
                    println!("invalid track duration 0");
                    return Err(AvifError::BmffParseFailed);
                }
                let remainder = if self.track_duration % self.segment_duration != 0 {
                    1u64
                } else {
                    0u64
                };
                let repetition_count: u64 =
                    (self.track_duration / self.segment_duration) + remainder - 1u64;
                if repetition_count > (i32::MAX as u64) {
                    return Ok(-1);
                } else {
                    return Ok(repetition_count as i32);
                }
            }
        }
        return Ok(0);
    }
}

#[derive(Debug)]
#[allow(unused)]
pub struct TimeToSample {
    pub sample_count: u32,
    pub sample_delta: u32,
}

#[derive(Debug)]
pub struct SampleToChunk {
    pub first_chunk: u32,
    pub samples_per_chunk: u32,
    #[allow(unused)]
    pub sample_description_index: u32,
}

#[derive(Debug, Default)]
pub struct SampleDescription {
    pub format: String,
    pub properties: Vec<ItemProperty>,
}

#[derive(Debug)]
pub enum SampleSize {
    FixedSize(u32),
    Sizes(Vec<u32>),
}

impl Default for SampleSize {
    fn default() -> Self {
        Self::FixedSize(0)
    }
}

#[derive(Debug, Default)]
pub struct SampleTable {
    pub chunk_offsets: Vec<u64>,
    pub sample_to_chunk: Vec<SampleToChunk>,
    pub sample_size: SampleSize,
    pub sync_samples: Vec<u32>,
    pub time_to_sample: Vec<TimeToSample>,
    pub sample_descriptions: Vec<SampleDescription>,
}

impl SampleTable {
    pub fn has_av1_sample(&self) -> bool {
        self.sample_descriptions.iter().any(|x| x.format == "av01")
    }

    // returns the number of samples in the chunk.
    pub fn get_sample_count_of_chunk(&self, chunk_index: u32) -> u32 {
        for entry in self.sample_to_chunk.iter().rev() {
            if entry.first_chunk <= chunk_index + 1 {
                return entry.samples_per_chunk;
            }
        }
        0
    }

    pub fn get_properties(&self) -> Option<&Vec<ItemProperty>> {
        Some(
            &self
                .sample_descriptions
                .iter()
                .find(|x| x.format == "av01")?
                .properties,
        )
    }

    pub fn sample_size(&self, index: usize) -> AvifResult<usize> {
        usize_from_u32(match &self.sample_size {
            SampleSize::FixedSize(size) => *size,
            SampleSize::Sizes(sizes) => {
                if index >= sizes.len() {
                    println!("not enough sampel sizes in the table");
                    return Err(AvifError::BmffParseFailed);
                }
                sizes[index]
            }
        })
    }
}
