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

// This module contains functions and types that will be exposed in the C API header file.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

pub mod decoder;
#[cfg(feature = "encoder")]
pub mod encoder;
pub mod gainmap;
pub mod image;
pub mod io;
pub mod reformat;
pub mod types;

/// Dereferences a non-null `*const T` into `&T`.
///
/// # Safety (macro caller invariant)
/// `$ptr` must be non-null (verified via `check_pointer!` / `check_pointer_or_return!` or an
/// explicit `.is_null()` check prior to invoking this macro), properly aligned, pointing to an
/// initialized `T`, and valid for shared reads without mutable aliasing for the lifetime of the
/// returned reference.
#[macro_export]
macro_rules! deref_const {
    ($ptr:expr) => {{
        // The extra curly braces make this whole macro into a single expression, and binding
        // `ptr` ensures `$ptr` is evaluated only once.
        let ptr = $ptr;
        assert!(!ptr.is_null());
        // SAFETY: `ptr` is non-null (verified by `assert!` above), and the macro caller upholds
        // that `ptr` is properly aligned, initialized, and valid for shared reads without mutable
        // aliasing.
        unsafe { &*ptr }
    }};
}

/// Dereferences a non-null `*mut T` into `&mut T`.
///
/// # Safety (macro caller invariant)
/// `$ptr` must be non-null (verified via `check_pointer!` / `check_pointer_or_return!` or an
/// explicit `.is_null()` check prior to invoking this macro), properly aligned, pointing to an
/// initialized `T`, and uniquely valid for reads and writes without any aliasing references for
/// the lifetime of the returned mutable reference.
#[macro_export]
macro_rules! deref_mut {
    ($ptr:expr) => {{
        // The extra curly braces make this whole macro into a single expression, and binding
        // `ptr` ensures `$ptr` is evaluated only once.
        let ptr = $ptr;
        assert!(!ptr.is_null());
        // SAFETY: `ptr` is non-null (verified by `assert!` above), and the macro caller upholds
        // that `ptr` is properly aligned, initialized, and uniquely valid for exclusive mutable
        // access.
        unsafe { &mut *ptr }
    }};
}

#[macro_export]
macro_rules! check_pointer {
    ($ptr:expr) => {
        if ($ptr).is_null() {
            return avifResult::InvalidArgument;
        }
    };
}

#[macro_export]
macro_rules! check_pointer_or_return {
    ($ptr:expr) => {
        if ($ptr).is_null() {
            return;
        }
    };
}
