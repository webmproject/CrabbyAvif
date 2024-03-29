use crate::internal_utils::*;
use crate::parser::mp4box::ItemProperty;
use crate::parser::mp4box::MetaBox;
use crate::*;

#[derive(Debug, Default)]
pub enum RepetitionCount {
    #[default]
    Unknown,
    Infinite,
    Finite(i32),
}

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
    pub fn check_limits(&self, size_limit: u32, dimension_limit: u32) -> bool {
        check_limits(self.width, self.height, size_limit, dimension_limit)
    }
    pub fn is_aux(&self, primary_track_id: u32) -> bool {
        if self.sample_table.is_none() || self.id == 0 {
            return false;
        }
        let sample_table = self.sample_table.unwrap_ref();
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

    pub fn repetition_count(&self) -> AvifResult<RepetitionCount> {
        if !self.elst_seen {
            return Ok(RepetitionCount::Unknown);
        }
        if self.is_repeating {
            if self.track_duration == u64::MAX {
                // If isRepeating is true and the track duration is unknown/indefinite, then set the
                // repetition count to infinite(Section 9.6.1 of ISO/IEC 23008-12 Part 12).
                return Ok(RepetitionCount::Infinite);
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
                let remainder =
                    if self.track_duration % self.segment_duration != 0 { 1u64 } else { 0u64 };
                let repetition_count: u64 =
                    (self.track_duration / self.segment_duration) + remainder - 1u64;
                return match i32::try_from(repetition_count) {
                    Ok(value) => Ok(RepetitionCount::Finite(value)),
                    Err(_) => Ok(RepetitionCount::Infinite),
                };
            }
        }
        Ok(RepetitionCount::Finite(0))
    }

    pub fn image_timing(&self, image_index: u32) -> AvifResult<ImageTiming> {
        let sample_table = self.sample_table.as_ref().ok_or(AvifError::NoContent)?;
        let mut image_timing = ImageTiming {
            timescale: self.media_timescale as u64,
            pts_in_timescales: 0,
            ..ImageTiming::default()
        };
        for i in 0..image_index as usize {
            image_timing.pts_in_timescales += sample_table.image_delta(i) as u64;
        }
        image_timing.duration_in_timescales = sample_table.image_delta(image_index as usize) as u64;
        if image_timing.timescale > 0 {
            image_timing.pts =
                image_timing.pts_in_timescales as f64 / image_timing.timescale as f64;
            image_timing.duration =
                image_timing.duration_in_timescales as f64 / image_timing.timescale as f64;
        } else {
            image_timing.pts = 0.0;
            image_timing.duration = 0.0;
        }
        Ok(image_timing)
    }
}

#[derive(Debug)]
pub struct TimeToSample {
    pub sample_count: u32,
    pub sample_delta: u32,
}

#[derive(Debug)]
pub struct SampleToChunk {
    pub first_chunk: u32,
    pub samples_per_chunk: u32,
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

    pub fn image_delta(&self, index: usize) -> u32 {
        let mut max_index = 0;
        for (i, time_to_sample) in self.time_to_sample.iter().enumerate() {
            max_index += time_to_sample.sample_count;
            if index < max_index as usize || i == self.time_to_sample.len() - 1 {
                return time_to_sample.sample_delta;
            }
        }
        1
    }
}

/// cbindgen:rename-all=CamelCase
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct ImageTiming {
    pub timescale: u64,
    pub pts: f64,
    pub pts_in_timescales: u64,
    pub duration: f64,
    pub duration_in_timescales: u64,
}
