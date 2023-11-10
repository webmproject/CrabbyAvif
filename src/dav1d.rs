use crate::bindings::*;
use crate::decoder::AvifImage;
use std::mem::MaybeUninit;

#[derive(Debug, Default)]
pub struct Dav1d {
    context: Option<*mut Dav1dContext>,
    picture: Option<Dav1dPicture>,
}

unsafe extern "C" fn avif_dav1d_free_callback(buf: *const u8, cookie: *mut ::std::os::raw::c_void) {
    // do nothing. the data is owned by the decoder.
}

fn dav1d_error(err: u32) -> i32 {
    let e: i32 = err.try_into().unwrap();
    -1i32 * e
}

impl Dav1d {
    pub fn initialize(&mut self, operating_point: u8, all_layers: bool) -> bool {
        let mut settings_uninit: MaybeUninit<Dav1dSettings> = unsafe { MaybeUninit::uninit() };
        unsafe { dav1d_default_settings(settings_uninit.as_mut_ptr()) };
        let mut settings = unsafe { settings_uninit.assume_init() };
        settings.max_frame_delay = 1;
        settings.n_threads = 8;
        // settings.frame_size_limit = xx;
        settings.operating_point = operating_point as i32;
        settings.all_layers = if all_layers { 1 } else { 0 };
        //println!("{:#?}", settings);

        unsafe {
            let mut dec = MaybeUninit::uninit();
            let ret = dav1d_open(dec.as_mut_ptr(), &settings);
            if ret != 0 {
                // TODO: carry forward the error.
                return false;
            }
            self.context = Some(dec.assume_init());
        }
        true
    }

    pub fn get_next_image(
        &mut self,
        av1_payload: &[u8],
        image: &mut AvifImage,
        category: usize,
    ) -> bool {
        if self.context.is_none() {
            if !self.initialize(0, true) {
                return false;
            }
        }
        let mut got_picture = false;
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
                return false;
            }
            let mut next_frame: Dav1dPicture = std::mem::zeroed();
            loop {
                if !data.data.is_null() {
                    let res = dav1d_send_data(self.context.unwrap(), &mut data);
                    println!("dav1d_send_data returned {res}");
                    // TODO: need to handle the error macros better.
                    if res < 0 && res != dav1d_error(EAGAIN) {
                        dav1d_data_unref(&mut data);
                        return false;
                    }
                }

                let res = dav1d_get_picture(self.context.unwrap(), &mut next_frame);
                println!("dav1d_get_picture returned {res}");
                if res == dav1d_error(EAGAIN) {
                    // send more data.
                    if !data.data.is_null() {
                        continue;
                    }
                    return false;
                } else if res < 0 {
                    if !data.data.is_null() {
                        dav1d_data_unref(&mut data);
                    }
                    return false;
                } else {
                    // Got a picture.
                    // TODO: layer selection.
                    got_picture = true;
                    break;
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
                // handle alpha special case.
            }
        }

        let dav1d_picture = &self.picture.unwrap();
        if (category == 0) {
            // if image dimensinos/yuv format does not match, deallocate the image.
            image.width = dav1d_picture.p.w as u32;
            image.height = dav1d_picture.p.h as u32;
            image.depth = dav1d_picture.p.bpc as u8;

            image.yuv_format = dav1d_picture.p.layout as u8;
            let seq_hdr = unsafe { (*dav1d_picture.seq_hdr) };
            image.full_range = seq_hdr.color_range != 0;
            image.chroma_sample_position = seq_hdr.chr as u8;

            image.color_primaries = seq_hdr.pri as u16;
            image.transfer_characteristics = seq_hdr.trc as u16;
            image.matrix_coefficients = seq_hdr.mtrx as u16;

            // TODO: call image freeplanes.
            let plane_count = if image.yuv_format == 0 { 1 } else { 3 };
            for plane in 0usize..plane_count {
                image.yuv_planes[plane] = Some(dav1d_picture.data[plane] as *mut u8);
                let stride_index = if plane == 0 { 0 } else { 1 };
                image.yuv_row_bytes[plane] = dav1d_picture.stride[stride_index] as u32;
            }
            image.image_owns_yuv_planes = false;
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
            image.alpha_plane = Some(dav1d_picture.data[0] as *mut u8);
            image.alpha_row_bytes = dav1d_picture.stride[0] as u32;
            image.image_owns_alpha_plane = false;
            let seq_hdr = unsafe { (*dav1d_picture.seq_hdr) };
            image.full_range = seq_hdr.color_range != 0;
        }
        true
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
