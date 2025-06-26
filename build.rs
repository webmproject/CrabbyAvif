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

#[cfg(feature = "capi")]
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    #[cfg(feature = "libgav1")]
    {
        // libgav1 needs libstdc++ on *nix/windows and libc++ on mac.
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }
    #[cfg(feature = "capi")]
    {
        // Generate the C header.
        let crate_path = env!("CARGO_MANIFEST_DIR");
        let config = cbindgen::Config::from_root_or_default(crate_path);
        let header_file = PathBuf::from(crate_path).join("include/avif/avif.h");
        cbindgen::Builder::new()
            .with_crate(crate_path)
            .with_config(config.clone())
            .generate()
            .unwrap()
            .write_to_file(header_file);

        // Generate the libavif compatible C header. This is the same as the C++ header above but
        // with the following modifications:
        // * No namespace.
        // * All functions are #define'd without the "crabby_" prefix.
        // * All constants are #define'd without the "CRABBY_" prefix.
        config.namespace = None;
        config.after_includes = Some(AFTER_INCLUDES_NO_NAMESPACE.to_string());

        let function_redefinitions: String = LIBAVIF_COMPAT_FUNCTIONS
            .iter()
            .map(|s| format!("#define {} crabby_{}\n", s, s))
            .collect::<Vec<String>>()
            .join("");
        config
            .after_includes
            .as_mut()
            .unwrap()
            .push_str(&function_redefinitions);

        let constant_redefinitions: String = LIBAVIF_COMPAT_CONSTANTS
            .iter()
            .map(|s| format!("#define {} CRABBY_{}\n", s, s))
            .collect::<Vec<String>>()
            .join("");
        config
            .after_includes
            .as_mut()
            .unwrap()
            .push_str(&constant_redefinitions);

        let header_file = PathBuf::from(crate_path).join("include/avif/avif_compat.h");
        cbindgen::Builder::new()
            .with_crate(crate_path)
            .with_config(config)
            .generate()
            .unwrap()
            .write_to_file(header_file);
    }
}
