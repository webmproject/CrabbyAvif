use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

pub mod decoder;
pub mod utils;
mod mp4box;
mod stream;
mod bindings;
mod dav1d;

// use crate::decoder::*;
// use crate::mp4box::*;
// use crate::stream::*;


// // TODO: learn to store references in struct and change this to a vec reference.
// pub fn avifparse(data : Vec<u8>) -> bool {
//     let mut stream = IStream {
//         data : data,
//         offset : 0,
//     };
//     let avif_boxes = MP4Box::parse(&mut stream);
    
//     let avif_items = match construct_avif_items(&avif_boxes.meta) {
//         Ok(items) => items,
//         Err(err) => {
//             println!("failed to construct_avif_items: {err}");
//             return false;
//         }
//     };
//     println!("{:#?}", avif_items);


//     let primary_item = avif_items.get(&avif_boxes.meta.primary_item_id).unwrap();
//     let extent = &primary_item.extents[0];
//     let offset : usize = extent.offset.try_into().unwrap();
//     let length : usize = extent.length.try_into().unwrap();
//     println!("offset: {} length: {}", offset, length);
//     let av1_payload = stream.data[offset..offset + length].to_vec().into_boxed_slice();
//     println!("av1 payload size: {}", av1_payload.len());
//     decode(av1_payload);
//     true
// }

// // TODO: need not return Vec<u8>, can just return c pointers. Memory lifetimes are still the same as libavif.
// pub fn decode_old(av1_payload : Box<[u8]>) -> [Vec<u8>; 3] {
//     let mut decoder = dav1d::Decoder::new().unwrap();
//     decoder.send_data(av1_payload, None, None, None);
//     let picture = loop {
//         match decoder.get_picture() {
//             Ok(p) => break p,
//             Err(err) => continue,
//         }
//     };
//     println!("{:#?}", picture);
//     let width : usize = picture.inner.pic.p.w.try_into().unwrap();
//     let height : usize = picture.inner.pic.p.h.try_into().unwrap();
//     println!("w: {width} h: {height}");

//     let filename = "/tmp/test.yuv";
//     let mut file = File::create(filename).unwrap();

//     // TODO: account for YUV subsampling.
//     let planes : [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
//     let mut ffmpeg_pixel_format = "yuv420p";
//     for plane in 0usize..3usize {
//         let ptr = picture.inner.pic.data[plane] as *const u8;
//         if ptr.is_null() {
//             ffmpeg_pixel_format = "gray";
//             break;
//         }
//         let stride_index : usize;
//         let plane_height : usize;
//         let plane_width : usize;
//         if plane == 0 {
//             stride_index = 0;
//             plane_height = height;
//             plane_width = width;
//         } else {
//             stride_index = 1;
//             if picture.inner.pic.p.layout == 1 {
//                 // 420.
//                 plane_height = (height + 1) / 2;
//                 plane_width = (width + 1) / 2;
//             } else if picture.inner.pic.p.layout == 2 {
//                 // 422
//                 plane_height = height;
//                 plane_width = (width + 1) / 2;
//             } else {
//                 plane_height = height;
//                 plane_width = width;
//             }
//         }
//         let stride : usize = picture.inner.pic.stride[stride_index].try_into().unwrap();
//         println!("plane {plane} isnull?: {} stride: {stride}", ptr.is_null());
//         let pixel_count : usize = (height * stride).try_into().unwrap();
//         for y in 0..plane_height {
//             let stride_offset : isize = (y * stride).try_into().unwrap();
//             let offset_ptr = unsafe { ptr.offset(stride_offset) };
//             let pixel_slice = unsafe { std::slice::from_raw_parts(offset_ptr, plane_width) };
//             match file.write_all(pixel_slice) {
//                 Err(e) => {
//                     println!("{:#?}", e);
//                     panic!("writing failed.");
//                 }
//                 _ => {},
//             };
//         }
//     }
//     let pix_fmts = ["gray", "yuv420p", "yuv422p", "yuv444p"];
//     let pix_fmt_index : usize = picture.inner.pic.p.layout.try_into().unwrap();
//     let pix_fmt = pix_fmts[pix_fmt_index];
//     println!("Wrote {filename}. dim: {width}x{height}");
//     println!("ffmpeg -s {width}x{height} -pix_fmt {pix_fmt} -f rawvideo -i {filename} -frames:v 1 -y /tmp/test.png");
//     planes
// }
