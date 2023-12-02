use crate::decoder::usize_from_u32;
use crate::parser::mp4box::ItemProperty;
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
    pub repetition_count: i32,
    pub width: u32,
    pub height: u32,
    pub sample_table: Option<SampleTable>,
    pub elst_seen: bool,
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

    // TODO: repetition count can be moved here.
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
