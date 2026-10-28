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

//! Error type.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The data ended before the syntax element could be read.
    Truncated,
    /// The bitstream violates a spec requirement.
    Invalid(&'static str),
    /// A valid bitstream feature this decoder does not support (e.g. inter
    /// frames).
    Unsupported(&'static str),
    /// Invalid decode options.
    InvalidOptions(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "truncated bitstream"),
            Error::Invalid(s) => write!(f, "invalid bitstream: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported feature: {s}"),
            Error::InvalidOptions(s) => write!(f, "invalid options: {s}"),
        }
    }
}

impl std::error::Error for Error {}
