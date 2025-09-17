// Copyright 2025 Google LLC
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

use crabby_avif::*;

mod utils;
use utils::*;

#[test]
fn decode_extended_pixi() -> AvifResult<()> {
    let mut decoder = get_decoder("extended_pixi.avif");
    assert_eq!(decoder.parse(), Ok(()));
    if !HAS_DECODER {
        return Ok(());
    }
    assert_eq!(decoder.next_image(), Ok(()));
    Ok(())
}
