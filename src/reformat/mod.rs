#[cfg(feature = "libyuv")]
pub mod alpha;
#[cfg(feature = "libyuv")]
pub mod libyuv;
#[cfg(feature = "libyuv")]
pub mod scale;

pub mod coeffs;
pub mod rgb;
pub mod rgb_impl;

// If libyuv is not present, add placeholder functions so that the library will build successfully
// without it.
#[cfg(not(feature = "libyuv"))]
pub mod libyuv {
    use crate::reformat::*;
    use crate::*;

    pub fn yuv_to_rgb(_image: &image::Image, _rgb: &mut rgb::Image) -> AvifResult<bool> {
        return Err(AvifError::NotImplemented);
    }

    pub fn process_alpha(_rgb: &mut rgb::Image, _multiply: bool) -> AvifResult<()> {
        return Err(AvifError::NotImplemented);
    }

    pub fn convert_to_half_float(_rgb: &mut rgb::Image, _scale: f32) -> AvifResult<()> {
        return Err(AvifError::NotImplemented);
    }

    impl image::Image {
        pub fn alpha_to_full_range(&mut self) -> AvifResult<()> {
            return Err(AvifError::NotImplemented);
        }
        pub fn scale(&mut self, _width: u32, _height: u32) -> AvifResult<()> {
            return Err(AvifError::NotImplemented);
        }
    }

    impl rgb::Image {
        pub fn premultiply_alpha(&mut self) -> AvifResult<()> {
            return Err(AvifError::NotImplemented);
        }
        pub fn unpremultiply_alpha(&mut self) -> AvifResult<()> {
            return Err(AvifError::NotImplemented);
        }
        pub fn set_opaque(&mut self) -> AvifResult<()> {
            return Err(AvifError::NotImplemented);
        }
        pub fn import_alpha_from(&mut self, _image: &image::Image) -> AvifResult<()> {
            return Err(AvifError::NotImplemented);
        }
    }
}
