pub mod bindings;

#[cfg(feature = "dav1d")]
pub mod dav1d;

#[cfg(feature = "libgav1")]
pub mod libgav1;

use crate::image::Image;
use crate::AvifResult;

pub trait Decoder {
    fn initialize(&mut self, operating_point: u8, all_layers: bool) -> AvifResult<()>;
    fn get_next_image(
        &mut self,
        av1_payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: usize,
    ) -> AvifResult<()>;
    // Destruction must be implemented using Drop.
}
