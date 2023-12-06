use crate::codecs::bindings::dav1d::*;
use crate::codecs::Decoder;
use crate::image::Image;
use crate::*;

use std::mem::MaybeUninit;

#[derive(Debug, Default)]
pub struct Dav1d {
    context: Option<*mut Dav1dContext>,
    picture: Option<Dav1dPicture>,
}

unsafe extern "C" fn avif_dav1d_free_callback(
    _buf: *const u8,
    _cookie: *mut ::std::os::raw::c_void,
) {
    // Do nothing. The buffers are owned by the decoder.
}

fn dav1d_error(err: u32) -> i32 {
    let e: i32 = err.try_into().unwrap();
    -e
}

impl Decoder for Dav1d {
    fn initialize(&mut self, operating_point: u8, all_layers: bool) -> AvifResult<()> {
        if self.context.is_some() {
            return Ok(()); // Already initialized.
        }
        let mut settings_uninit: MaybeUninit<Dav1dSettings> = MaybeUninit::uninit();
        unsafe { dav1d_default_settings(settings_uninit.as_mut_ptr()) };
        let mut settings = unsafe { settings_uninit.assume_init() };
        settings.max_frame_delay = 1;
        settings.n_threads = 8;
        // settings.frame_size_limit = xx;
        settings.operating_point = operating_point as i32;
        settings.all_layers = if all_layers { 1 } else { 0 };

        unsafe {
            let mut dec = MaybeUninit::uninit();
            let ret = dav1d_open(dec.as_mut_ptr(), &settings);
            if ret != 0 {
                // TODO: carry forward the error within the enum as a string.
                // Here and elsewhere in this file.
                return Err(AvifError::UnknownError);
            }
            self.context = Some(dec.assume_init());
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
        if self.context.is_none() {
            self.initialize(0, true)?;
        }
        let got_picture;
        let av1_payload_len = av1_payload.len();
        println!("paylaoad len: {av1_payload_len}");
        unsafe {
            let mut data: Dav1dData = std::mem::zeroed();
            let res = dav1d_data_wrap(
                &mut data,
                av1_payload.as_ptr(),
                av1_payload_len,
                Some(avif_dav1d_free_callback),
                /*cookie=*/ std::ptr::null_mut(),
            );
            println!("dav1d_data_wrap returned {res}");
            if res != 0 {
                return Err(AvifError::UnknownError);
            }
            let mut next_frame: Dav1dPicture = std::mem::zeroed();
            loop {
                if !data.data.is_null() {
                    let res = dav1d_send_data(self.context.unwrap(), &mut data);
                    println!("dav1d_send_data returned {res}");
                    // TODO: need to handle the error macros better.
                    if res < 0 && res != dav1d_error(EAGAIN) {
                        dav1d_data_unref(&mut data);
                        return Err(AvifError::UnknownError);
                    }
                }

                let res = dav1d_get_picture(self.context.unwrap(), &mut next_frame);
                println!("dav1d_get_picture returned {res}");
                if res == dav1d_error(EAGAIN) {
                    // send more data.
                    if !data.data.is_null() {
                        continue;
                    }
                    return Err(AvifError::UnknownError);
                } else if res < 0 {
                    if !data.data.is_null() {
                        dav1d_data_unref(&mut data);
                    }
                    return Err(AvifError::UnknownError);
                } else {
                    // Got a picture.
                    let frame_spatial_id = (*next_frame.frame_hdr).spatial_id as u8;
                    if spatial_id != 0xFF && spatial_id != frame_spatial_id {
                        // layer selection: skip this unwanted layer.
                        dav1d_picture_unref(&mut next_frame);
                    } else {
                        got_picture = true;
                        break;
                    }
                }
            }
            if !data.data.is_null() {
                dav1d_data_unref(&mut data);
            }

            if got_picture {
                // unref previous frame.
                if self.picture.is_some() {
                    dav1d_picture_unref(&mut self.picture.unwrap());
                }
                self.picture = Some(next_frame);
                // store other fields like color range, etc.
            } else {
                // TODO: handle alpha special case.
            }
        }

        let dav1d_picture = &self.picture.unwrap();
        if category == 0 {
            // if image dimensinos/yuv format does not match, deallocate the image.
            image.width = dav1d_picture.p.w as u32;
            image.height = dav1d_picture.p.h as u32;
            image.depth = dav1d_picture.p.bpc as u8;

            image.yuv_format = match dav1d_picture.p.layout {
                0 => PixelFormat::Monochrome,
                1 => PixelFormat::Yuv420,
                2 => PixelFormat::Yuv422,
                3 => PixelFormat::Yuv444,
                _ => PixelFormat::Yuv420, // not reached.
            };
            let seq_hdr = unsafe { *dav1d_picture.seq_hdr };
            image.full_range = seq_hdr.color_range != 0;
            image.chroma_sample_position = seq_hdr.chr.into();

            image.color_primaries = (seq_hdr.pri as u16).into();
            image.transfer_characteristics = (seq_hdr.trc as u16).into();
            image.matrix_coefficients = (seq_hdr.mtrx as u16).into();

            // TODO: call image freeplanes.
            for plane in 0usize..image.yuv_format.plane_count() {
                image.planes[plane] = Some(dav1d_picture.data[plane] as *const u8);
                let stride_index = if plane == 0 { 0 } else { 1 };
                image.row_bytes[plane] = dav1d_picture.stride[stride_index] as u32;
            }
            image.image_owns_planes = false;
        } else if category == 1 {
            /*
            if image.width != (dav1d_picture.p.w as u32)
                || image.height != (dav1d_picture.p.h as u32)
                || image.depth != (dav1d_picture.p.bpc as u8)
            {
                // Alpha plane does not match the previous alpha plane.
                return false;
            }
            */
            image.width = dav1d_picture.p.w as u32;
            image.height = dav1d_picture.p.h as u32;
            image.depth = dav1d_picture.p.bpc as u8;
            // TODO: call image freeplanes.
            image.planes[3] = Some(dav1d_picture.data[0] as *const u8);
            image.row_bytes[3] = dav1d_picture.stride[0] as u32;
            image.image_owns_alpha_plane = false;
            let seq_hdr = unsafe { *dav1d_picture.seq_hdr };
            image.full_range = seq_hdr.color_range != 0;
        }
        // TODO: gainmap category.
        Ok(())
    }
}

impl Drop for Dav1d {
    fn drop(&mut self) {
        if self.picture.is_some() {
            println!("unreffing dav1d picture");
            unsafe { dav1d_picture_unref(&mut self.picture.unwrap()) };
        }
        if self.context.is_some() {
            println!("closing dav1d");
            unsafe { dav1d_close(&mut self.context.unwrap()) };
        }
    }
}