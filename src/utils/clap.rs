use crate::internal_utils::*;

#[derive(Debug, Clone, Copy)]
#[allow(unused)]
pub struct CleanAperture {
    pub width: UFraction,
    pub height: UFraction,
    pub horiz_off: UFraction,
    pub vert_off: UFraction,
}
