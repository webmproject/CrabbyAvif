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

//! Constant tables from the AV1 specification.
//!
//! OWNER: tables worker (except where noted). Naming: spec table names in
//! SCREAMING_SNAKE_CASE (e.g. `Num_4x4_Blocks_Wide` -> `NUM_4X4_BLOCKS_WIDE`),
//! same dimensions and index order as the spec.
//!
//! The table data was extracted mechanically from the spec text. Element
//! types are the smallest integer type that fits the values, except where
//! the spec element is a block size / transform size / transform type /
//! intra mode (stored as the [`crate::decoder::av1::types`] enums) or a 0/1 flag
//! (stored as `bool`).

pub mod block;
pub mod cdf;
pub mod misc;
pub mod qm;
pub mod quant;
pub mod scan;
