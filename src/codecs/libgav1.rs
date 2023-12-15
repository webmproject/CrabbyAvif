use crate::codecs::bindings::libgav1::*;
use crate::codecs::Decoder;
use crate::image::Image;
use crate::*;

use std::mem::MaybeUninit;

#[derive(Debug, Default)]
pub struct Libgav1 {
    decoder: Option<*mut Libgav1Decoder>,
    image: Option<Libgav1DecoderBuffer>,
}

#[allow(non_upper_case_globals)]
impl Decoder for Libgav1 {
    fn initialize(&mut self, operating_point: u8, all_layers: bool) -> AvifResult<()> {
        if self.decoder.is_some() {
            return Ok(()); // Already initialized.
        }
        let mut settings_uninit: MaybeUninit<Libgav1DecoderSettings> = MaybeUninit::uninit();
        unsafe {
            Libgav1DecoderSettingsInitDefault(settings_uninit.as_mut_ptr());
        }
        let mut settings = unsafe { settings_uninit.assume_init() };
        settings.threads = 8;
        settings.operating_point = operating_point as i32;
        settings.output_all_layers = if all_layers { 1 } else { 0 };
        unsafe {
            let mut dec = MaybeUninit::uninit();
            let ret = Libgav1DecoderCreate(&settings, dec.as_mut_ptr());
            if ret != Libgav1StatusCode_kLibgav1StatusOk {
                // TODO: carry forward the error within the enum as a string.
                // Here and elsewhere in this file.
                return Err(AvifError::UnknownError);
            }
            self.decoder = Some(dec.assume_init());
        }
        Ok(())
    }

    fn get_next_image(
        &mut self,
        av1_payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: usize,
    ) -> AvifResult<()> {
        if self.decoder.is_none() {
            self.initialize(0, true)?;
        }
        unsafe {
            let ret = Libgav1DecoderEnqueueFrame(
                self.decoder.unwrap(),
                av1_payload.as_ptr(),
                av1_payload.len(),
                0,
                std::ptr::null_mut(),
            );
            if ret != Libgav1StatusCode_kLibgav1StatusOk {
                println!("enqueue failed. err: {ret} len: {}", av1_payload.len());
                return Err(AvifError::UnknownError);
            }
            self.image = None;
            let mut next_frame: *const Libgav1DecoderBuffer = std::ptr::null_mut();
            loop {
                let ret = Libgav1DecoderDequeueFrame(self.decoder.unwrap(), &mut next_frame);
                if ret != Libgav1StatusCode_kLibgav1StatusOk {
                    println!("dequeue failed. err: {ret}");
                    return Err(AvifError::UnknownError);
                }
                if !next_frame.is_null()
                    && spatial_id != 0xFF
                    && (*next_frame).spatial_id as u8 != spatial_id
                {
                    next_frame = std::ptr::null_mut();
                } else {
                    break;
                }
            }
            // Got an image.
            if next_frame.is_null() {
                if category == 1 {
                    // TODO: handle alpha special case.
                } else {
                    println!("next frame is null. err: {ret}");
                    return Err(AvifError::UnknownError);
                }
            } else {
                self.image = Some(*next_frame);
                // TODO: store color range.
            }

            let gav1_image = &self.image.unwrap();
            if category == 0 {
                image.width = gav1_image.displayed_width[0] as u32;
                image.height = gav1_image.displayed_height[0] as u32;
                image.depth = gav1_image.bitdepth as u8;

                image.yuv_format = match gav1_image.image_format {
                    Libgav1ImageFormat_kLibgav1ImageFormatMonochrome400 => PixelFormat::Monochrome,
                    Libgav1ImageFormat_kLibgav1ImageFormatYuv420 => PixelFormat::Yuv420,
                    Libgav1ImageFormat_kLibgav1ImageFormatYuv422 => PixelFormat::Yuv422,
                    Libgav1ImageFormat_kLibgav1ImageFormatYuv444 => PixelFormat::Yuv444,
                    _ => PixelFormat::Yuv420, // not reached.
                };
                image.full_range =
                    gav1_image.color_range != Libgav1ColorRange_kLibgav1ColorRangeStudio;
                image.chroma_sample_position = gav1_image.chroma_sample_position.into();

                image.color_primaries = (gav1_image.color_primary as u16).into();
                image.transfer_characteristics =
                    (gav1_image.transfer_characteristics as u16).into();
                image.matrix_coefficients = (gav1_image.matrix_coefficients as u16).into();

                // TODO: call free planes.
                for plane in 0usize..image.yuv_format.plane_count() {
                    image.planes[plane] = Some(gav1_image.plane[plane] as *mut u8);
                    image.row_bytes[plane] = gav1_image.stride[plane] as u32;
                    image.image_owns_planes[plane] = false;
                }
            } else if category == 1 {
                // TODO: make sure alpha plane matches previous alpha plane.
                image.width = gav1_image.displayed_width[0] as u32;
                image.height = gav1_image.displayed_height[0] as u32;
                image.depth = gav1_image.bitdepth as u8;
                // TODO: call image freeplanes.
                image.planes[3] = Some(gav1_image.plane[0] as *mut u8);
                image.row_bytes[3] = gav1_image.stride[0] as u32;
                image.image_owns_planes[3] = false;
                image.full_range =
                    gav1_image.color_range != Libgav1ColorRange_kLibgav1ColorRangeStudio;
            }
            // TODO: gainmap category.
        }
        Ok(())
    }
}

impl Drop for Libgav1 {
    fn drop(&mut self) {
        if self.decoder.is_some() {
            println!("closing gav1");
            unsafe { Libgav1DecoderDestroy(self.decoder.unwrap()) };
        }
    }
}
